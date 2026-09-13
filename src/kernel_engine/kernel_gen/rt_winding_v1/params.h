#pragma once
#include <optix.h>
#include <cuda_runtime.h>
struct WindingParams {
    OptixTraversableHandle gas;
    const float3* corners;
    const float3* rays;
    int* winding;
};
