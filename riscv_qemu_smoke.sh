#!/usr/bin/env bash
set -euo pipefail

# Usage: ./benchmarks/riscv_qemu_smoke.sh <input.s> [rv32|rv64]
asm_file=${1:-}
arch=${2:-rv64}

if [[ -z "$asm_file" ]]; then
  echo "Usage: $0 <input.s> [rv32|rv64]" >&2
  exit 1
fi

base="${asm_file%.*}"
out="${base}.elf"

case "$arch" in
  rv32)
    target=riscv32-unknown-elf
    qemu=qemu-system-riscv32
    ;;
  rv64)
    target=riscv64-unknown-elf
    qemu=qemu-system-riscv64
    ;;
  *)
    echo "Unknown arch: $arch (expected rv32|rv64)" >&2
    exit 1
    ;;
esac

${target}-gcc -static -nostartfiles -Wl,-e,main -o "$out" "$asm_file"

echo "Running under $qemu: $out"
# Use system emulator with minimal kernel setup
$qemu -machine virt -cpu rv64 -kernel "$out" -nographic


