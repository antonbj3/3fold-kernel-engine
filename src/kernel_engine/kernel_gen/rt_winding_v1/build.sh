#!/usr/bin/env bash
set -euo pipefail
: "${OPTIX_INCLUDE:?set external SDK include directory}"
: "${CUDA_ROOT:?set external CUDA toolkit directory}"
: "${RT_BUILD_DIR:?set output directory outside source tree}"
rt_source=$(cd -- "$(dirname -- "$0")" && pwd)
mkdir -p "$RT_BUILD_DIR"
"$CUDA_ROOT/bin/nvcc" --ptx -std=c++17 -arch=compute_75 --fmad=false -I"$OPTIX_INCLUDE" "$rt_source/program.cu" -o "$RT_BUILD_DIR/program.ptx"
g++ -O2 -std=c++17 -I"$OPTIX_INCLUDE" -I"$CUDA_ROOT/include" "$rt_source/host.cpp" -L"$CUDA_ROOT/lib64" -Wl,-rpath,"$CUDA_ROOT/lib64" -lcudart -ldl -o "$RT_BUILD_DIR/rt_winding"
