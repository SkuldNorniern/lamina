"""Validate emitted kernels on gfx1200 devices through the HIP module API."""

import argparse
import ctypes as C
import os
import pathlib
import tempfile
import struct
import subprocess

root = pathlib.Path(__file__).resolve().parents[3]
rocm = pathlib.Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
parser = argparse.ArgumentParser(
    description="Execute AMDGPU fixtures on gfx1200 through HIP"
)
parser.add_argument("--opt-level", type=int, choices=range(4), default=0)
parser.add_argument("--compiler", default=str(root / "target/debug/lamina"))
parser.add_argument("--clang", default="clang")
parser.add_argument("--linker", default=str(rocm / "llvm/bin/ld.lld"))
parser.add_argument("--hip-library", default=str(rocm / "lib/libamdhip64.so"))
options = parser.parse_args()
with tempfile.TemporaryDirectory(prefix="lamina-amdgpu-runtime-") as directory:
    work = pathlib.Path(directory)
    subprocess.run(
        [
            options.compiler,
            str(pathlib.Path(__file__).with_name("amdgpu.lamina")),
            "--target",
            "amdgcn_amdhsa",
            "--mcpu",
            "gfx1200",
            "--opt-level",
            str(options.opt_level),
            "--emit-asm",
            "-o",
            str(work / "runtime"),
        ],
        check=True,
    )
    subprocess.run(
        [
            options.clang,
            "--target=amdgcn-amd-amdhsa",
            "-mcpu=gfx1200",
            "-c",
            str(work / "runtime.s"),
            "-o",
            str(work / "runtime.o"),
        ],
        check=True,
    )
    subprocess.run(
        [
            options.linker,
            "-shared",
            str(work / "runtime.o"),
            "-o",
            str(work / "runtime.hsaco"),
        ],
        check=True,
    )
    hip = C.CDLL(options.hip_library)
    P = C.c_void_p
    U = C.c_uint
    I = C.c_int
    S = C.c_size_t
    signatures = {
        "hipInit": [U],
        "hipGetDeviceCount": [C.POINTER(I)],
        "hipSetDevice": [I],
        "hipModuleLoad": [C.POINTER(P), C.c_char_p],
        "hipModuleUnload": [P],
        "hipModuleGetFunction": [C.POINTER(P), P, C.c_char_p],
        "hipModuleLaunchKernel": [
            P,
            U,
            U,
            U,
            U,
            U,
            U,
            U,
            P,
            C.POINTER(P),
            C.POINTER(P),
        ],
        "hipDeviceSynchronize": [],
        "hipMalloc": [C.POINTER(P), S],
        "hipFree": [P],
        "hipMemcpyHtoD": [P, P, S],
        "hipMemcpyDtoH": [P, P, S],
    }
    for name, args in signatures.items():
        fn = getattr(hip, name)
        fn.argtypes = args
        fn.restype = I
    hip.hipGetErrorString.argtypes = [I]
    hip.hipGetErrorString.restype = C.c_char_p

    def call(name, *args):
        result = getattr(hip, name)(*args)
        if result:
            raise RuntimeError(
                f"{name}: {hip.hipGetErrorString(result).decode()} ({result})"
            )

    call("hipInit", 0)
    count = I()
    call("hipGetDeviceCount", C.byref(count))
    if count.value == 0:
        raise RuntimeError("No HIP devices available")
    print("HIP devices:", count.value, flush=True)
    for device in range(count.value):
        call("hipSetDevice", device)
        module = P()
        call("hipModuleLoad", C.byref(module), str(work / "runtime.hsaco").encode())
        allocations = []

        def allocate(data):
            ptr = P()
            call("hipMalloc", C.byref(ptr), len(data))
            allocations.append(ptr)
            host = C.create_string_buffer(data)
            call("hipMemcpyHtoD", ptr, host, len(data))
            return ptr

        def read(ptr, size):
            host = C.create_string_buffer(size)
            call("hipMemcpyDtoH", host, ptr, size)
            return host.raw

        def launch(name, arguments=(), grid=(1, 1, 1), block=(1, 1, 1)):
            function = P()
            call("hipModuleGetFunction", C.byref(function), module, name.encode())
            parameters = (P * len(arguments))(
                *(C.cast(C.pointer(arg), P) for arg in arguments)
            )
            call(
                "hipModuleLaunchKernel",
                function,
                *grid,
                *block,
                0,
                None,
                parameters,
                None,
            )
            call("hipDeviceSynchronize")

        out = allocate(bytes(192 * 8))
        for a, b in [(0, 0), (1, 2), (-1, 1), (-2147483648, -1), (2147483647, 15)]:
            launch("arithmetic32", (out, I(a), I(b)))
            expected = [
                a + b,
                a - b,
                a * b,
                a & b,
                a | b,
                a ^ b,
                a << 3,
                (b & 0xFFFFFFFF) >> 4,
                a >> 31,
            ]
            actual = struct.unpack("<9I", read(out, 36))
            assert actual == tuple(v & 0xFFFFFFFF for v in expected), (
                device,
                "i32",
                a,
                b,
                actual,
                expected,
            )
        for a, b in [
            (0, 0),
            (1, 2),
            (0xFFFFFFFFFFFFFFFF, 1),
            (0xFFFFFFFF, 0x100000001),
            (0x8123456789ABCDEF, 0xFEDCBA9876543210),
        ]:
            launch("arithmetic64", (out, C.c_uint64(a), C.c_uint64(b)))
            expected = [
                a + b,
                a - b,
                a * b,
                a & b,
                a | b,
                a ^ b,
                a + 0xFFFFFFFF,
                a << 3,
                b >> 4,
                (a if a < 1 << 63 else a - (1 << 64)) >> 31,
            ]
            actual = struct.unpack("<10Q", read(out, 80))
            assert actual == tuple(v & 0xFFFFFFFFFFFFFFFF for v in expected), (
                device,
                "i64",
                a,
                b,
                actual,
                expected,
            )
        for a, b in [(-1, 0xFEDCBA9876543210), (-2147483648, 0x100000001), (123, 456)]:
            launch("conversions", (out, I(a), C.c_uint64(b), out))
            data = read(out, 32)
            actual = (
                struct.unpack_from("<Q", data, 0)[0],
                struct.unpack_from("<Q", data, 8)[0],
                struct.unpack_from("<I", data, 16)[0],
                struct.unpack_from("<Q", data, 24)[0],
            )
            expected = (
                a & 0xFFFFFFFF,
                a & 0xFFFFFFFFFFFFFFFF,
                b & 0xFFFFFFFF,
                out.value,
            )
            assert actual == expected, (device, "conversions", actual, expected)
        left = [i * 10000001 - 1000000000 for i in range(128)]
        right = [2000000000 - i * 20000003 for i in range(128)]
        lp = allocate(struct.pack("<128i", *left))
        rp = allocate(struct.pack("<128i", *right))
        launch("vector_add", (out, lp, rp, I(64)), grid=(2, 1, 1), block=(64, 1, 1))
        actual = struct.unpack("<128I", read(out, 512))
        assert actual == tuple((a + b) & 0xFFFFFFFF for a, b in zip(left, right)), (
            device,
            "vector_add",
            actual,
        )
        left64 = [
            (i * 1234567 << 32) | (0xFFFFFFFF if i % 2 else i) for i in range(128)
        ]
        right64 = [((128 - i) << 32) | (i + 1) for i in range(128)]
        lp64 = allocate(struct.pack("<128Q", *left64))
        rp64 = allocate(struct.pack("<128Q", *right64))
        for name, expected in [
            (
                "vector_add64",
                [(a + b) & 0xFFFFFFFFFFFFFFFF for a, b in zip(left64, right64)],
            ),
            (
                "vector_mul64",
                [(a * b) & 0xFFFFFFFFFFFFFFFF for a, b in zip(left64, right64)],
            ),
        ]:
            launch(name, (out, lp64, rp64, I(64)), grid=(2, 1, 1), block=(64, 1, 1))
            actual = struct.unpack("<128Q", read(out, 1024))
            assert actual == tuple(expected), (device, name, actual, expected)
        inp = allocate(struct.pack("<Q", 0xFFFFFFFFFFFFFFFF))
        launch("load_store64", (out, inp))
        assert read(out, 8) == bytes(8), (device, "load_store64")
        launch("coordinates", (out,), grid=(2, 2, 2), block=(4, 3, 2))
        expected = []
        for gz in range(2):
            for gy in range(2):
                for gx in range(2):
                    for z in range(2):
                        for y in range(3):
                            for x in range(4):
                                expected.append(
                                    x
                                    | (y << 4)
                                    | (z << 8)
                                    | (gx << 16)
                                    | (gy << 20)
                                    | (gz << 24)
                                )
        actual = struct.unpack("<192I", read(out, 768))
        assert actual == tuple(expected), (device, "coordinates", actual, expected)
        for grid, block in [((3, 2, 4), (5, 3, 2)), ((1, 1, 1), (1, 1, 1))]:
            records = grid[0] * grid[1] * grid[2] * block[0] * block[1] * block[2]
            dimensions = allocate(bytes(records * 32))
            launch("dimensions", (dimensions, C.c_uint64(0xFFFFFFFF)), grid=grid, block=block)
            expected_record = struct.pack("<6IQ", *block, *grid, 0x100000000)
            assert read(dimensions, records * 32) == expected_record * records, (device, "dimensions", grid, block)
        launch("dimensions_without_kernarg", grid=(2, 3, 4), block=(3, 2, 1))
        launch("barrier", grid=(2, 1, 1), block=(64, 1, 1))
        for pointer in allocations:
            call("hipFree", pointer)
        call("hipModuleUnload", module)
        print(
            f"device {device}: arithmetic, conversions, vector add, 64-bit memory, 3D IDs, dispatch dimensions, barrier passed",
            flush=True,
        )
