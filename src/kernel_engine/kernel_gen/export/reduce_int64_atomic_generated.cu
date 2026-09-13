#define WP_NO_BFLOAT16

#define WP_TILE_BLOCK_DIM 256
// EXPORT EDIT: WP_NO_CRT removed -- offline nvcc has the real CRT headers (NVRTC does not)
#include "builtin.h"
#include "deterministic.h"

// Map wp.breakpoint() to a device brkpt at the call site so cuda-gdb attributes the stop to the generated .cu line
#if defined(__CUDACC__) && !defined(_MSC_VER)
#define __debugbreak() __brkpt()
#endif

#define builtin_tid1d() wp::tid(_idx, dim)
#define builtin_tid2d(x, y) wp::tid(x, y, _idx, dim)
#define builtin_tid3d(x, y, z) wp::tid(x, y, z, _idx, dim)
#define builtin_tid4d(x, y, z, w) wp::tid(x, y, z, w, _idx, dim)

#define builtin_block_dim() wp::block_dim()

// CUDA Thread Block Cluster shape declaration. Expands to __cluster_dims__
// only on devices that support clusters (compute capability 9.0+); otherwise
// expands to nothing so the same source compiles cleanly for any target arch.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define WP_CLUSTER_DIMS(x, y, z) __cluster_dims__(x, y, z)
#else
#define WP_CLUSTER_DIMS(x, y, z)
#endif

// Maximum registers per thread. __maxnreg__ was added in CUDA Toolkit 12.4;
// older toolkits ignore the opt-in so the same source remains compilable.
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 4))
#define WP_MAXNREG(n) __maxnreg__(n)
#else
#define WP_MAXNREG(n)
#endif

// Allow PTXAS to spill registers into shared memory. CUDA Toolkit 13.0
// introduced the pragma, which is unavailable in device-debug compilation.
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 13) && !defined(_DEBUG)
#define WP_ENABLE_SMEM_SPILLING() asm volatile(".pragma \"enable_smem_spilling\";");
#else
#define WP_ENABLE_SMEM_SPILLING()
#endif


// avoid namespacing of float type for casting to float type, this is to avoid wp::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

#define int(x) cast_int(x)
#define adj_int(x, adj_x, adj_ret) adj_cast_int(x, adj_x, adj_ret)



extern "C" __global__ void reduce_int64_atomic_7863aa6b_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::int32> var_tgt,
    wp::array_t<wp::int64> var_qval,
    wp::array_t<wp::int64> var_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32* var_1;
        wp::int64* var_2;
        wp::int64 var_3;
        wp::int32 var_4;
        wp::int64 var_5;
        //---------
        // forward
        // def reduce_int64_atomic(tgt: wp.array(dtype=wp.int32), qval: wp.array(dtype=wp.int64), out: wp.array(dtype=wp.int64)):       <L 10>
        // i = wp.tid()                                                                           <L 11>
        var_0 = builtin_tid1d();
        // wp.atomic_add(out, tgt[i], qval[i])                                                    <L 12>
        var_1 = wp::address(var_tgt, var_0);
        var_2 = wp::address(var_qval, var_0);
        var_4 = wp::load(var_1);
        var_5 = wp::load(var_2);
        var_3 = wp::atomic_add(var_out, var_4, var_5);
    }
}



extern "C" __global__ void reduce_int64_atomic_7863aa6b_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::int32> var_tgt,
    wp::array_t<wp::int64> var_qval,
    wp::array_t<wp::int64> var_out,
    wp::array_t<wp::int32> adj_tgt,
    wp::array_t<wp::int64> adj_qval,
    wp::array_t<wp::int64> adj_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32* var_1;
        wp::int64* var_2;
        wp::int64 var_3;
        wp::int32 var_4;
        wp::int64 var_5;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int64 adj_2 = {};
        wp::int64 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int64 adj_5 = {};
        //---------
        // forward
        // def reduce_int64_atomic(tgt: wp.array(dtype=wp.int32), qval: wp.array(dtype=wp.int64), out: wp.array(dtype=wp.int64)):       <L 10>
        // i = wp.tid()                                                                           <L 11>
        var_0 = builtin_tid1d();
        // wp.atomic_add(out, tgt[i], qval[i])                                                    <L 12>
        var_1 = wp::address(var_tgt, var_0);
        var_2 = wp::address(var_qval, var_0);
        var_4 = wp::load(var_1);
        var_5 = wp::load(var_2);
        // var_3 = wp::atomic_add(var_out, var_4, var_5);
        //---------
        // reverse
        wp::adj_atomic_add(var_out, var_4, var_5, adj_out, adj_1, adj_2, adj_3);
        wp::adj_address(var_qval, var_0, adj_qval, adj_0, adj_2);
        wp::adj_address(var_tgt, var_0, adj_tgt, adj_0, adj_1);
        // adj: wp.atomic_add(out, tgt[i], qval[i])                                               <L 12>
        // adj: i = wp.tid()                                                                      <L 11>
        // adj: def reduce_int64_atomic(tgt: wp.array(dtype=wp.int32), qval: wp.array(dtype=wp.int64), out: wp.array(dtype=wp.int64)):  <L 10>
        continue;
    }
}

