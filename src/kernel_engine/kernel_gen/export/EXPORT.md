# Warp -> CUDA C++ export: the int64 fixed-point reduction

Question: a kernel written and verified in Warp, exported as CUDA C++ source and compiled with `nvcc`
into a plain C++ host program (no Warp runtime, no Python) - does it keep (a) bit-identical results and
(b) its bandwidth fraction?

Answer on this machine (RTX 5070, sm_120, CUDA 12.8 `nvcc`, Warp 1.17.0, driver-reported toolkit 12.9):
**(a) yes, bit-identical; (b) yes, 11.1-11.2 % of the 632 GB/s identity under Warp vs 11.6 % under nvcc
(ratio 1.03-1.04, inside the 10 % band).**

Kernel: `reduce_int64_atomic` from `src/kernel_engine/reductions/u_v54_gather_law_deterministic_reduction_vs_atomic_int64.py`
- one thread per contribution, `wp.atomic_add(out, tgt[i], qval[i])` on `int64` fixed-point values
(scale `1<<30`), 200 000 contributions into 64 slots. Order-invariant because integer addition is
associative; that property is what has to survive the export.

## Export steps (exactly what was done)

1. Run the kernel once under Warp. Warp writes the generated CUDA C++ for the module into its kernel
   cache, one directory per module hash, containing `<module>.cu`, `<module>.meta` and the compiled
   `<module>.sm120.ptx`. The `.cu` holds both a `..._cuda_kernel_forward` and a `..._cuda_kernel_backward`
   entry point; the forward one is the kernel.
2. Copy that `.cu` to `reduce_int64_atomic_generated.cu`. One edit: the generated preamble's
   `#define WP_NO_CRT` is commented out. That macro tells Warp's `crt.h` to redeclare `int64_t`,
   `printf`, `assert` etc. because NVRTC has no C runtime headers; offline `nvcc` pre-includes
   `cuda_runtime.h` and therefore glibc's, and the two declarations collide. Removing the macro is the
   whole delta between the JIT path and the offline path.
3. Copy the header closure. `nvcc -M` on the generated file names 35 headers from the Warp package's
   `native/` directory (`builtin.h`, `crt.h`, `array.h`, `vec.h`, `mat.h`, `tile*.h`, `nanovdb/PNanoVDB.h`, ...).
   They are in `warp_native/`, unmodified, Apache-2.0 (see `THIRD_PARTY`). The exported tree needs no
   Warp installation after this step.
4. Compile with `-DWP_ENABLE_CUDA=1`. Without it `tile.h` takes its "no CUDA toolkit" branch and defines
   its own `float4`, which clashes with `vector_types.h`.

`build.sh` is the full command: `nvcc -O3 -std=c++17 -arch=sm_120a -DNDEBUG -DWP_ENABLE_CUDA=1
-diag-suppress 177,550 -Iwarp_native host_int64_reduce.cu export_shim.cu reduce_int64_atomic_generated.cu`.
It is warning-free and takes about 4 s.

## The shim (Warp's launch ABI)

The generated kernel is `extern "C" __global__` and takes Warp's launch descriptor by value:

```c++
extern "C" __global__ void reduce_int64_atomic_<hash>_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::int32> var_tgt, wp::array_t<wp::int64> var_qval, wp::array_t<wp::int64> var_out);
```

`export_shim.cu` is the only file that includes Warp headers. It fills those structs from plain pointers
and launches:

* `wp::launch_bounds_t<N>` = `{ int shape[N]; size_t size; size_t coord_mult; }`. For a 1-D launch:
  `shape[0] = n`, `size = n`, `coord_mult = 1`.
* `wp::array_t<T>` has a host-callable 1-D constructor `array_t(T* data, int size)` that sets
  `shape.dims[0]`, `ndim = 1`, `strides[0] = sizeof(T)`, `grad = nullptr`, `flags = 0`. Arrays are passed
  **by value**, so a foreign caller only needs the device pointer and the length.
* Launch configuration: this kernel body is a grid-stride loop, so Warp uses a 1-D grid,
  `block_dim = 256` (its default) and `grid = ceil(n / block_dim)` with no `max_blocks` cap. The shim
  reproduces exactly that; any grid that covers the loop gives the same result, only the timing moves.

The host program `host_int64_reduce.cu` uses only the CUDA runtime API: read the two `.npy` inputs,
`cudaMalloc`/`cudaMemcpy`, `cudaMemset` the 64 output slots, call the shim, copy back, print the sum and
an FNV-1a hash of the raw output bytes, then time 20 repetitions with CUDA events after a 10-launch
warm-up. No Warp runtime symbol is needed at link time: everything the generated code calls
(`wp::address`, `wp::load`, `wp::atomic_add`, `wp::tid`) is header-inline and device-side.

## Numbers

Same input for both sides: seed 0, `tgt` (int32, 200 000 values in [0,64)) and `qval`
(int64 = round(float32 value * 2^30)) written to `.npy` by the Warp driver and read back by the host
program. Bytes moved per launch = 200 000*4 + 200 000*8 + 64*8*2 = 2 401 024 B.
Timing = CUDA events, 20 repetitions after warm-up, kernel only (no allocation, no H2D, no memset);
machine at `nice -n 15`, GPU otherwise idle (1.6 GiB resident desktop, 6 % utilisation).

| path | run | sum of slots | FNV-1a of output bytes | us/launch | GB/s | fraction of 632 GB/s identity |
|---|---|---|---|---|---|---|
| Warp-launched   | 1 | 25737627924 | 2925d09e14ba | 34.09 | 70.4 | 0.111 |
| Warp-launched   | 2 | 25737627924 | 2925d09e14ba | 33.90 | 70.8 | 0.112 |
| exported + nvcc | 1 | 25737627924 | 2925d09e14ba | 32.82 | 73.2 | 0.116 |
| exported + nvcc | 2 | 25737627924 | 2925d09e14ba | 32.89 | 73.0 | 0.116 |

(a) Bit-identity: all 64 int64 slots compare equal element-wise between the Warp output and the exported
output, and the hash is the same across both runs of both paths. The order-invariance the kernel was
selected for is a property of the emitted code, not of the launcher.

(b) Bandwidth: exported/Warp = 1.04 and 1.03, i.e. the exported path is marginally *faster* - consistent
with dropping Warp's per-launch Python and argument-packing overhead on a 33 us kernel - and well inside
the 10 % band. The absolute fraction, 0.11 of the identity, is the kernel's own ceiling: 200 000 atomics
into 64 slots is contention-bound, not bandwidth-bound; that is a property of the algorithm and is
identical on both paths, which is the point of the comparison.

## What a foreign engine needs to call this

A plain C or C++ engine that wants this kernel without Warp:

1. Ship `reduce_int64_atomic_generated.cu` + `warp_native/` (1.5 MB of headers, Apache-2.0) and compile
   them into the engine with the flags above, or compile once to a `.cubin`/`.fatbin` and load it with
   the driver API by the `extern "C"` name.
2. Call through a C entry point like the one in `export_shim.cu`:
   `void wp_export_launch_reduce_int64(const int32_t* tgt, const int64_t* qval, int64_t* out, int n, int nb, int block_dim, cudaStream_t)`.
   That signature is C-callable and carries no Warp types, so it can be declared in a C header and
   called from a C engine, from Rust/ctypes, or from another scheduler.
3. If the engine cannot compile C++ at all, the two structs have to be mirrored in C, because they are
   passed by value as kernel arguments: `struct { int shape[1]; size_t size; size_t coord_mult; }` and
   `struct { void* data; void* grad; int dims[4]; int strides[4]; int ndim; uint32_t flags; }` - check
   the layout of `wp::array_t<T>` in `warp_native/array.h` against the toolkit in use before relying on
   it; the shim approach avoids this and is the recommended route.
4. Memory ownership stays with the engine: the kernel reads and writes raw device pointers only, so
   Warp's allocator, streams and mempool are not involved.

Limits of this result: one kernel, one arch (`sm_120a`), one toolkit. Kernels that use Warp's
higher-level device features (meshes, BVHs, hash grids, volumes, tiles) call into structures that the
Warp *runtime* builds on the host; for those the headers alone are not enough and the host side would
have to construct the same descriptors. This kernel touches none of them.

## Reproduce

```
python3 warp_reference_run.py <outdir>     # writes in_tgt.npy, in_qval.npy, out_warp.npy; prints WARP line
./build.sh <builddir>
<builddir>/host_int64_reduce <outdir>      # prints NVCC line, writes out_nvcc.bin
```
Compare `out_warp.npy` with `out_nvcc.bin` (64 int64) - they must be equal element-wise.
`tests/test_kernel_gen_export.py` does exactly this and skips when `nvcc` or a CUDA device is missing.
