#!/usr/bin/env bash
# Build the exported kernel into a plain C++/CUDA host program. No Warp runtime, no Python.
#   ./build.sh [outdir]        -> <outdir>/host_int64_reduce
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HERE/build}"
mkdir -p "$OUT"
nvcc -O3 -std=c++17 -arch=sm_120a -DNDEBUG -DWP_ENABLE_CUDA=1 -diag-suppress 177,550 \
     -I"$HERE/warp_native" \
     -o "$OUT/host_int64_reduce" \
     "$HERE/host_int64_reduce.cu" "$HERE/export_shim.cu" "$HERE/reduce_int64_atomic_generated.cu"
echo "built $OUT/host_int64_reduce"
