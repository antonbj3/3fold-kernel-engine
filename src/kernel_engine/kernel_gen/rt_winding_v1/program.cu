#include "params.h"
extern "C" { __constant__ WindingParams params; }
extern "C" __global__ void __raygen__winding() {
    const unsigned i = optixGetLaunchIndex().x;
    unsigned sum = 0;
    optixTrace(params.gas, params.rays[i], make_float3(0, 0, 1),
               0.f, 1.e16f, 0.f, 255, OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
               0, 1, 0, sum);
    params.winding[i] = static_cast<int>(sum);
}
extern "C" __global__ void __miss__winding() {}
extern "C" __global__ void __anyhit__winding() {
    if (optixGetRayTmax() > 0.f) {
        const unsigned face = optixGetPrimitiveIndex();
        const float3 a = params.corners[3*face];
        const float3 b = params.corners[3*face+1];
        const float3 c = params.corners[3*face+2];
        const double nz = (double(b.x)-a.x)*(double(c.y)-a.y)
                        -(double(b.y)-a.y)*(double(c.x)-a.x);
        const int sign = (nz > 0) - (nz < 0);
        optixSetPayload_0(optixGetPayload_0() + static_cast<unsigned>(sign));
    }
    optixIgnoreIntersection();
}
