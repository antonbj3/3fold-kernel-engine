// extern "C" shim: builds Warp's launch ABI (launch_bounds_t + array_t) around the generated kernel.
// This is the only translation unit that includes Warp headers; the host program is Warp-free.
#include "builtin.h"
#include <cuda_runtime.h>
#include <cstdint>

extern "C" __global__ void reduce_int64_atomic_7863aa6b_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::int32> var_tgt,
    wp::array_t<wp::int64> var_qval,
    wp::array_t<wp::int64> var_out);

extern "C" void wp_export_launch_reduce_int64(
    const int32_t* d_tgt, const int64_t* d_qval, int64_t* d_out,
    int n, int nb, int block_dim, cudaStream_t stream)
{
    wp::launch_bounds_t<1> bounds;
    bounds.shape[0] = n;
    bounds.size = (size_t)n;
    bounds.coord_mult = 1;                       // 1 thread per launch coord (no tile/block multiplicity)
    wp::array_t<wp::int32> a_tgt((wp::int32*)d_tgt, n);
    wp::array_t<wp::int64> a_qval((wp::int64*)d_qval, n);
    wp::array_t<wp::int64> a_out((wp::int64*)d_out, nb);
    int grid = (n + block_dim - 1) / block_dim;  // grid-stride kernel, max_blocks unset -> natural grid
    reduce_int64_atomic_7863aa6b_cuda_kernel_forward<<<grid, block_dim, 0, stream>>>(bounds, a_tgt, a_qval, a_out);
}
