// Native allocation and C ABI around the unchanged generated forward kernel.
#include "builtin.h"
#include "stream_api.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" __global__ void stream_3bfa5cbb_cuda_kernel_forward(
    wp::launch_bounds_t<2>, wp::array_t<wp::float32>, wp::array_t<wp::float32>,
    wp::array_t<wp::int32>, wp::array_t<wp::float32>, wp::array_t<wp::float32>,
    wp::array_t<wp::int32>, wp::int32, wp::int32);

static void check(cudaError_t result) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(result));
        throw result;
    }
}

struct Storage {
    float *source=nullptr, *out=nullptr, *cx=nullptr, *cy=nullptr;
    int32_t *mask=nullptr, *op=nullptr;
    cudaEvent_t begin=nullptr, end=nullptr;
    ~Storage() {
        cudaFree(source); cudaFree(out); cudaFree(cx); cudaFree(cy);
        cudaFree(mask); cudaFree(op);
        if (begin) cudaEventDestroy(begin);
        if (end) cudaEventDestroy(end);
    }
};

extern "C" int stream_export_run(const float *source, const int32_t *mask,
    const float *cx, const float *cy, const int32_t *op, int n,
    float *output, double *event_ms, double *wall_ms) {
    if (n != 512 && n != 1024) return 2;
    Storage s;
    try {
        const size_t cells=size_t(n)*n, bytes=9*cells*sizeof(float);
        check(cudaMalloc(&s.source,bytes)); check(cudaMalloc(&s.out,bytes));
        check(cudaMalloc(&s.mask,cells*sizeof(int32_t)));
        check(cudaMalloc(&s.cx,9*sizeof(float))); check(cudaMalloc(&s.cy,9*sizeof(float)));
        check(cudaMalloc(&s.op,9*sizeof(int32_t)));
        check(cudaMemcpy(s.source,source,bytes,cudaMemcpyHostToDevice));
        check(cudaMemcpy(s.mask,mask,cells*sizeof(int32_t),cudaMemcpyHostToDevice));
        check(cudaMemcpy(s.cx,cx,9*sizeof(float),cudaMemcpyHostToDevice));
        check(cudaMemcpy(s.cy,cy,9*sizeof(float),cudaMemcpyHostToDevice));
        check(cudaMemcpy(s.op,op,9*sizeof(int32_t),cudaMemcpyHostToDevice));
        wp::launch_bounds_t<2> bounds;
        bounds.shape[0]=n; bounds.shape[1]=n; bounds.size=cells; bounds.coord_mult=1;
        wp::array_t<wp::float32> a(s.source,9,n,n), b(s.out,9,n,n), x(s.cx,9), y(s.cy,9);
        wp::array_t<wp::int32> m(s.mask,n,n), o(s.op,9);
        auto launch=[&]() {
            stream_3bfa5cbb_cuda_kernel_forward<<<(cells+255)/256,256>>>(bounds,a,b,m,x,y,o,n,n);
            check(cudaGetLastError());
        };
        launch(); check(cudaMemcpy(output,s.out,bytes,cudaMemcpyDeviceToHost));
        std::vector<float> repeat(9*cells);
        launch(); check(cudaMemcpy(repeat.data(),s.out,bytes,cudaMemcpyDeviceToHost));
        if (std::memcmp(output,repeat.data(),bytes)) return 3;
        for (int i=0;i<10;++i) launch();
        check(cudaDeviceSynchronize());
        check(cudaEventCreate(&s.begin)); check(cudaEventCreate(&s.end));
        const auto start=std::chrono::steady_clock::now();
        check(cudaEventRecord(s.begin));
        for (int i=0;i<30;++i) launch();
        check(cudaEventRecord(s.end)); check(cudaDeviceSynchronize());
        *wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/30;
        float elapsed=0; check(cudaEventElapsedTime(&elapsed,s.begin,s.end));
        *event_ms=elapsed/30.0;
        return 0;
    } catch (cudaError_t) { return 4; }
}
