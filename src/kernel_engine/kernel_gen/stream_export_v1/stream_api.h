#ifndef STREAM_EXPORT_API_H
#define STREAM_EXPORT_API_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int stream_export_run(const float *source, const int32_t *mask,
                      const float *cx, const float *cy, const int32_t *op,
                      int n, float *output, double *event_ms, double *wall_ms);
#ifdef __cplusplus
}
#endif
#endif
