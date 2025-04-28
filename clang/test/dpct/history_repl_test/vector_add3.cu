// RUN: echo "empty command"

#include <cuda.h>
#include <stdio.h>


// CHECK: void VectorAddKernel(float* A, float* B, float* C)
// CHECK-NEXT: {
// CHECK-NEXT: #ifdef _FOO_
// CHECK-NEXT:      A[threadIdx.x] = threadIdx.x + 4.0f;
// CHECK-NEXT: #else
// CHECK-NEXT:      auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
// CHECK-NEXT:      A[item_ct1.get_local_id(2)] = item_ct1.get_local_id(2) + 5.0f;
// CHECK-NEXT: #endif 
// CHECK-NEXT: }
__global__ void VectorAddKernel(float* A, float* B, float* C)
{
#ifdef _FOO_
     A[threadIdx.x] = threadIdx.x + 4.0f;
#else
     A[threadIdx.x] = threadIdx.x + 5.0f;
#endif 
}
