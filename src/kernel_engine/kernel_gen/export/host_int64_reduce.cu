// Minimal CUDA-runtime host program that launches the Warp-generated kernel. No Warp runtime.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

extern "C" void wp_export_launch_reduce_int64(const int32_t* d_tgt, const int64_t* d_qval,
                                              int64_t* d_out, int n, int nb, int block_dim, cudaStream_t stream);

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} } while(0)

static std::vector<char> read_npy(const char* path, size_t& nbytes_out) {
    FILE* f = fopen(path, "rb"); if(!f){fprintf(stderr,"open %s\n",path);exit(1);}
    char magic[8];
    unsigned short hlen;
    if (fread(magic,1,8,f)!=8) { fprintf(stderr,"short read (magic) %s\n",path); exit(1); }
    if (fread(&hlen,2,1,f)!=1) { fprintf(stderr,"short read (header len) %s\n",path); exit(1); }
    std::vector<char> hdr(hlen);
    if (fread(hdr.data(),1,hlen,f)!=hlen) { fprintf(stderr,"short read (header) %s\n",path); exit(1); }
    long cur = ftell(f); fseek(f,0,SEEK_END); long end = ftell(f); fseek(f,cur,SEEK_SET);
    nbytes_out = (size_t)(end-cur);
    std::vector<char> buf(nbytes_out);
    if (fread(buf.data(),1,nbytes_out,f)!=nbytes_out) { fprintf(stderr,"short read (data) %s\n",path); exit(1); }
    fclose(f);
    return buf;
}

int main(int argc, char** argv) {
    const char* dir = argv[1];
    char p1[512], p2[512]; snprintf(p1,sizeof p1,"%s/in_tgt.npy",dir); snprintf(p2,sizeof p2,"%s/in_qval.npy",dir);
    size_t nb_t, nb_q;
    std::vector<char> tgt = read_npy(p1, nb_t);
    std::vector<char> qv  = read_npy(p2, nb_q);
    const int N = (int)(nb_t/4);
    const int NB = 64;
    if (nb_q != (size_t)N*8) { fprintf(stderr,"size mismatch\n"); return 1; }

    int32_t* d_tgt; int64_t* d_qv; int64_t* d_out;
    CK(cudaMalloc(&d_tgt, nb_t)); CK(cudaMalloc(&d_qv, nb_q)); CK(cudaMalloc(&d_out, NB*8));
    CK(cudaMemcpy(d_tgt, tgt.data(), nb_t, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_qv, qv.data(), nb_q, cudaMemcpyHostToDevice));
    CK(cudaMemset(d_out, 0, NB*8));

    const int block = 256;   // Warp's default block_dim

    wp_export_launch_reduce_int64(d_tgt, d_qv, d_out, N, NB, block, 0);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    std::vector<int64_t> out(NB);
    CK(cudaMemcpy(out.data(), d_out, NB*8, cudaMemcpyDeviceToHost));
    long long sum = 0; for (int i=0;i<NB;i++) sum += out[i];
    // FNV-1a over the raw output bytes
    uint64_t h = 1469598103934665603ULL; const unsigned char* pb=(const unsigned char*)out.data();
    for (size_t i=0;i<NB*8;i++){ h ^= pb[i]; h *= 1099511628211ULL; }

    char po[512]; snprintf(po,sizeof po,"%s/out_nvcc.bin",dir);
    FILE* fo = fopen(po,"wb"); fwrite(out.data(),8,NB,fo); fclose(fo);

    // timing: warm-up 10, then 20 timed reps (kernel only, as on the Warp side)
    for (int i=0;i<10;i++) wp_export_launch_reduce_int64(d_tgt, d_qv, d_out, N, NB, block, 0);
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int REPS=20;
    cudaEventRecord(e0);
    for (int i=0;i<REPS;i++) wp_export_launch_reduce_int64(d_tgt, d_qv, d_out, N, NB, block, 0);
    cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
    float ms=0; cudaEventElapsedTime(&ms,e0,e1); ms /= REPS;
    double bytes = (double)N*4 + (double)N*8 + (double)NB*8*2;
    printf("NVCC sum=%lld hash=%012llx ms=%.6f GBs=%.1f\n", sum, (unsigned long long)(h & 0xffffffffffffULL), ms, bytes/ms*1e-6);
    return 0;
}
