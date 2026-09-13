// C ABI instrumentation for the unmodified upstream kernel.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include "vectorAdd.cu"
extern "C" int run_foreign(const float* a,const float* b,float* out,int n,int repeats,float* ms,float* copy_ms) {
    if(n<0 || repeats<1) return -1;
    float *da=nullptr,*db=nullptr,*dc=nullptr;
    cudaEvent_t start=nullptr,stop=nullptr;
    int status=0;
    size_t bytes=(size_t(n)+32)*sizeof(float);
    const int blocks=std::max(1,(n+255)/256);
    #define CHECK(call) do {cudaError_t e=(call);if(e!=cudaSuccess){status=int(e);goto cleanup;}} while(0)
    CHECK(cudaMalloc(&da,bytes));CHECK(cudaMalloc(&db,bytes));CHECK(cudaMalloc(&dc,bytes));
    CHECK(cudaMemcpy(da,a,bytes,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db,b,bytes,cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dc,0x5a,bytes));
    CHECK(cudaEventCreate(&start));CHECK(cudaEventCreate(&stop));
    for(int i=0;i<5;i++) vecAdd<<<blocks,256>>>(da,db,dc,n);
    CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(start));
    for(int i=0;i<repeats;i++) vecAdd<<<blocks,256>>>(da,db,dc,n);
    CHECK(cudaGetLastError());CHECK(cudaEventRecord(stop));CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(ms,start,stop));*ms/=repeats;
    CHECK(cudaMemcpy(out,dc,bytes,cudaMemcpyDeviceToHost));
    CHECK(cudaEventRecord(start));
    for(int i=0;i<repeats;i++) CHECK(cudaMemcpyAsync(dc,da,bytes,cudaMemcpyDeviceToDevice));
    CHECK(cudaEventRecord(stop));CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(copy_ms,start,stop));*copy_ms/=repeats;
    cleanup:
    if(start) cudaEventDestroy(start);if(stop) cudaEventDestroy(stop);
    if(da) cudaFree(da);if(db) cudaFree(db);if(dc) cudaFree(dc);
    return status;
}
