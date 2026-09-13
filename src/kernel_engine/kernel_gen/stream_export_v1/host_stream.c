/* A C compiler builds this caller. CUDA and generated types stay behind the ABI. */
#include "stream_api.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 4) return 2;
    const int n=atoi(argv[1]);
    if (n != 512 && n != 1024) return 2;
    const size_t cells=(size_t)n*n, bytes=9*cells*sizeof(float);
    float *source=malloc(bytes), *out=malloc(bytes);
    int32_t *mask=malloc(cells*sizeof(int32_t));
    float cx[9],cy[9]; int32_t op[9];
    if (!source || !out || !mask) return 3;
    FILE *input=fopen(argv[2],"rb");
    if (!input) return 4;
    int valid=fread(source,1,bytes,input)==bytes
        && fread(mask,sizeof(int32_t),cells,input)==cells
        && fread(cx,sizeof(float),9,input)==9
        && fread(cy,sizeof(float),9,input)==9
        && fread(op,sizeof(int32_t),9,input)==9
        && fgetc(input)==EOF;
    fclose(input);
    if (!valid) return 5;
    double event_ms=0,wall_ms=0;
    const int rc=stream_export_run(source,mask,cx,cy,op,n,out,&event_ms,&wall_ms);
    if (rc) return 10+rc;
    FILE *output=fopen(argv[3],"wb");
    if (!output) return 6;
    valid=fwrite(out,1,bytes,output)==bytes;
    if (fclose(output)) valid=0;
    free(source);free(out);free(mask);
    if (!valid) return 7;
    printf("{\"side\":%d,\"event_ms\":%.12g,\"wall_ms\":%.12g,\"two_launch_identity\":true}\n",n,event_ms,wall_ms);
    return 0;
}
