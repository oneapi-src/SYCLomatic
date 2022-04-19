// RUN: dpct --format-range=none -out-root=%T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: dpct --format-range=none -out-root=%T %S/vector_add3.cu --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: dpct --format-range=none -out-root=%T -extra-arg="-D_FOO_" %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only

// RUN: FileCheck --input-file %T/vector_add.dp.cpp --match-full-lines %s
// RUN: FileCheck --input-file %T/vector_add3.dp.cpp --match-full-lines %S/vector_add3.cu

#include <cuda.h>
#include <stdio.h>

// CHECK: void VectorAddKernel(float* A, float* B, float* C, sycl::nd_item<3> item_ct1)
// CHECK-NEXT: {
// CHECK-NEXT: #ifdef _FOO_
// CHECK-NEXT:      A[item_ct1.get_local_id(2)] = item_ct1.get_local_id(2) + 4.0f;
// CHECK-NEXT: #else
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

