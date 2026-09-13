#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:?build directory required}"
ARCH="${2:-sm_89}"
mkdir -p "$OUT"
gcc -O3 -std=c11 -Wall -Wextra -Werror -I"$HERE/../stream_export_v1" -c "$HERE/../stream_export_v1/host_stream.c" -o "$OUT/host_stream.o"
nvcc -O3 -std=c++17 -arch="$ARCH" -DNDEBUG -DWP_ENABLE_CUDA=1 -diag-suppress 177,550 \
    -I"$HERE/../export/warp_native" -I"$HERE/../stream_export_v1" \
    "$OUT/host_stream.o" "$HERE/stream_shim.cu" "$HERE/../stream_export_v1/stream_generated.cu" -o "$OUT/host_stream"
