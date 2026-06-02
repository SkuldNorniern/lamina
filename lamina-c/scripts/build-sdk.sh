#!/usr/bin/env bash
# build-sdk.sh — assemble a lamina-c release archive.
#
# Usage:
#   ./scripts/build-sdk.sh [--release] [--nightly] [--out DIR]
#
# Outputs:
#   <out>/lamina-c-<version>-<target>.tar.gz
#   <out>/lamina-c-<version>-<target>.tar.gz.sha256
#
# Must be run from the lamina-c/ directory.

set -euo pipefail

PROFILE=debug
NIGHTLY=0
OUT_DIR="$(pwd)/dist"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --release)   PROFILE=release ;;
        --nightly)   NIGHTLY=1 ;;
        --out=*)     OUT_DIR="${1#*=}" ;;
        --out)       shift; OUT_DIR="$1" ;;
        *)           echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
    shift
done

CARGO_FLAGS=()
[ "$PROFILE" = "release" ] && CARGO_FLAGS+=(--release)
[ "$NIGHTLY"  = "1" ]      && CARGO_FLAGS+=(--features nightly)

# ---------------------------------------------------------------------------
# Detect version and target
# ---------------------------------------------------------------------------

VERSION=$(cargo metadata --no-deps --format-version 1 | \
    python3 -c "import sys,json; d=json.load(sys.stdin); \
    print(next(p['version'] for p in d['packages'] if p['name']=='lamina-c'))")

ARCH=$(uname -m)
OS_RAW=$(uname -s | tr '[:upper:]' '[:lower:]')
case "$OS_RAW" in
    linux*)   OS=linux ;;
    darwin*)  OS=macos ;;
    msys*|mingw*|cygwin*) OS=windows ;;
    *)        OS="$OS_RAW" ;;
esac
TARGET="${ARCH}-${OS}"

CHANNEL=stable
[ "$NIGHTLY" = "1" ] && CHANNEL=nightly

ARCHIVE_NAME="lamina-c-${VERSION}-${CHANNEL}-${TARGET}"
ARCHIVE_DIR="${OUT_DIR}/${ARCHIVE_NAME}"

echo "==> lamina-c ${VERSION} | ${CHANNEL} | ${TARGET}"
echo "    profile : ${PROFILE}"
echo "    out     : ${ARCHIVE_DIR}"

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

echo "==> cargo build"
cargo build "${CARGO_FLAGS[@]}"

CARGO_TARGET_DIR="$(cargo metadata --no-deps --format-version 1 | \
    python3 -c "import sys,json; print(json.load(sys.stdin)['target_directory'])")"
LIB_DIR="${CARGO_TARGET_DIR}/${PROFILE}"

# ---------------------------------------------------------------------------
# Assemble archive
# ---------------------------------------------------------------------------

rm -rf "$ARCHIVE_DIR"
mkdir -p \
    "${ARCHIVE_DIR}/include" \
    "${ARCHIVE_DIR}/lib" \
    "${ARCHIVE_DIR}/examples" \
    "${ARCHIVE_DIR}/pkgconfig"

# Headers
cp include/lamina.h "${ARCHIVE_DIR}/include/"
[ "$NIGHTLY" = "1" ] && cp include/lamina_nightly.h "${ARCHIVE_DIR}/include/"

# Libraries
case "$OS" in
    linux)
        [ -f "${LIB_DIR}/liblamina_c.so" ] && cp "${LIB_DIR}/liblamina_c.so"  "${ARCHIVE_DIR}/lib/"
        [ -f "${LIB_DIR}/liblamina_c.a"  ] && cp "${LIB_DIR}/liblamina_c.a"   "${ARCHIVE_DIR}/lib/"
        ;;
    macos)
        [ -f "${LIB_DIR}/liblamina_c.dylib" ] && cp "${LIB_DIR}/liblamina_c.dylib" "${ARCHIVE_DIR}/lib/"
        [ -f "${LIB_DIR}/liblamina_c.a"     ] && cp "${LIB_DIR}/liblamina_c.a"     "${ARCHIVE_DIR}/lib/"
        ;;
    windows)
        [ -f "${LIB_DIR}/lamina_c.dll"     ] && cp "${LIB_DIR}/lamina_c.dll"     "${ARCHIVE_DIR}/lib/"
        [ -f "${LIB_DIR}/lamina_c.dll.lib" ] && cp "${LIB_DIR}/lamina_c.dll.lib" "${ARCHIVE_DIR}/lib/"
        [ -f "${LIB_DIR}/lamina_c.lib"     ] && cp "${LIB_DIR}/lamina_c.lib"     "${ARCHIVE_DIR}/lib/"
        ;;
esac

# Examples
cp examples/builder_aot.c "${ARCHIVE_DIR}/examples/"
cp examples/raw_ir.c      "${ARCHIVE_DIR}/examples/"
[ "$NIGHTLY" = "1" ] && cp examples/builder_jit.c "${ARCHIVE_DIR}/examples/"

# pkg-config (Unix only)
if [ "$OS" != "windows" ]; then
    cat > "${ARCHIVE_DIR}/pkgconfig/lamina.pc" <<EOF
prefix=\${pcfiledir}/../..
includedir=\${prefix}/include
libdir=\${prefix}/lib

Name: lamina-c
Description: Lamina compiler C API
Version: ${VERSION}
Cflags: -I\${includedir}
Libs: -L\${libdir} -llamina_c
EOF
fi

# Metadata
echo "${VERSION}" > "${ARCHIVE_DIR}/VERSION"
cp ../LICENSE "${ARCHIVE_DIR}/LICENSE"

cat > "${ARCHIVE_DIR}/README.md" <<EOF
# lamina-c ${VERSION} — ${CHANNEL} — ${TARGET}

C API for the Lamina compiler. Header files are in \`include/\`.
Libraries are in \`lib/\`. See \`examples/\` for usage.

Documentation: https://github.com/SkuldNorniern/lamina/blob/main/docs/c-bindings.md

## Quick start (Linux/macOS)

\`\`\`sh
cc examples/builder_aot.c \\
    -I include -L lib -llamina_c \\
    -Wl,-rpath,\$(pwd)/lib \\
    -o builder_aot
./builder_aot
\`\`\`

## ABI version

C ABI: ${VERSION}
Channel: ${CHANNEL}
EOF

# ---------------------------------------------------------------------------
# Smoke test: compile builder_aot.c against packaged artifacts
# ---------------------------------------------------------------------------

echo "==> smoke test: compile against packaged artifacts"

RPATH_FLAG=""
[ "$OS" = "linux" ]  && RPATH_FLAG="-Wl,-rpath,$(realpath "${ARCHIVE_DIR}/lib")"
[ "$OS" = "macos" ]  && RPATH_FLAG="-Wl,-rpath,$(realpath "${ARCHIVE_DIR}/lib")"

cc examples/builder_aot.c \
    -I "${ARCHIVE_DIR}/include" \
    -L "${ARCHIVE_DIR}/lib" -llamina_c \
    ${RPATH_FLAG} \
    -o "${ARCHIVE_DIR}/smoke_test_builder_aot"

echo "    compile: ok"
"${ARCHIVE_DIR}/smoke_test_builder_aot" > /dev/null
echo "    run    : ok"
rm "${ARCHIVE_DIR}/smoke_test_builder_aot"

# ---------------------------------------------------------------------------
# Archive + checksum
# ---------------------------------------------------------------------------

mkdir -p "$OUT_DIR"
TARBALL="${OUT_DIR}/${ARCHIVE_NAME}.tar.gz"

echo "==> creating ${TARBALL}"
tar -czf "$TARBALL" -C "$OUT_DIR" "$ARCHIVE_NAME"

CHECKSUM_FILE="${TARBALL}.sha256"
sha256sum "$TARBALL" > "$CHECKSUM_FILE"
echo "==> sha256: $(cat "$CHECKSUM_FILE")"

echo ""
echo "Archive : ${TARBALL}"
echo "Checksum: ${CHECKSUM_FILE}"
echo "Contents:"
tar -tzf "$TARBALL" | sort | sed 's/^/  /'
