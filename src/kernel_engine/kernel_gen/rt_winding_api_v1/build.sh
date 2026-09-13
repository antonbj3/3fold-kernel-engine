#!/usr/bin/env bash
set -euo pipefail
: "${OPTIX_INCLUDE:?external SDK include required}"
: "${CUDA_ROOT:?external toolkit required}"
: "${RT_BUILD_DIR:?external output directory required}"
rt_source=$(cd -- "$(dirname -- "$0")" && pwd)
mkdir -p "$RT_BUILD_DIR"
g++ -shared -fPIC -pthread -O2 -std=c++17 -I"$OPTIX_INCLUDE" -I"$CUDA_ROOT/include" "$rt_source/api.cpp" -L"$CUDA_ROOT/lib64" -Wl,-rpath,"$CUDA_ROOT/lib64" -lcudart -ldl -o "$RT_BUILD_DIR/librt_winding_api.so"
