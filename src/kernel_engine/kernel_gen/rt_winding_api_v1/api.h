#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Same-thread use; immutable triangle corners float32[M,3,3], local coordinates.
 * Status: 0 success,1 invalid input,2 runtime error,3 wrong thread.
 * Caller provides valid buffers/handles; no concurrent use or context switching.
 * query writes int32[count] only on successful validated execution.
 * destroy nulls caller handle; a null handle is an idempotent success. */
int rt_create(const char* ptx_file,const float* corners,uint32_t triangles,uint32_t capacity,void** handle);
int rt_query(void* handle,const float* origins,uint32_t count,int32_t* output);
int rt_destroy(void** handle);
#ifdef __cplusplus
}
#endif
