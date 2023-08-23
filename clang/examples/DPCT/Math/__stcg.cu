#include "cuda_bf16.h"
#include "cuda_fp16.h"

__global__ void test(__half *ph, __half h, __half2 *ph2, __half2 h2,
                     __nv_bfloat16 *pb, __nv_bfloat16 b, __nv_bfloat162 *pb2,
                     __nv_bfloat162 b2) {
  // Start
  __stcg(ph /*__half **/, h /*__half*/);
  __stcg(ph2 /*__half2 **/, h2 /*__half2*/);
  __stcg(pb /*__nv_bfloat16 **/, b /*__nv_bfloat16*/);
  __stcg(pb2 /*__nv_bfloat162 **/, b2 /*__nv_bfloat162*/);
  // End
}
