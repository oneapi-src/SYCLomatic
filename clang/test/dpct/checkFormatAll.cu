// RUN: cat %s > %T/checkFormatAll.cu
// RUN: cd %T
// RUN: dpct -out-root %T/checkFormatAll checkFormatAll.cu --cuda-include-path="%cuda-path/include" --format-range=all -- -std=c++14  -x cuda --cuda-host-only
// RUN: FileCheck -strict-whitespace checkFormatAll.cu --match-full-lines --input-file %T/checkFormatAll/checkFormatAll.dp.cpp

#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include "cublas_v2.h"

     //CHECK:void testDevice(const int *K) { int t = K[0]; }
__device__ void testDevice(const int *K) {
  int t = K[0];
}

     //CHECK:void testDevice1(const int *K) { int t = K[0]; }
__device__ void testDevice1(const int *K) { int t = K[0]; }

     //CHECK:void testKernelPtr(const int *L, const int *M, int N,
//CHECK-NEXT:                   const sycl::nd_item<3> &item_ct1) {
//CHECK-NEXT:  testDevice(L);
//CHECK-NEXT:  int gtid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
//CHECK-NEXT:             item_ct1.get_local_id(2);
//CHECK-NEXT:}
__global__ void testKernelPtr(const int *L, const int *M, int N) {
  testDevice(L);
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
}


     //CHECK:int main() {
//CHECK-NEXT:  sycl::device dev_ct1{};
//CHECK-NEXT:  sycl::queue q_ct1(dev_ct1,
//CHECK-NEXT:                    sycl::property_list{sycl::property::queue::in_order()});
//CHECK-NEXT:  sycl::range<3> griddim = sycl::range<3>(1, 1, 2);
//CHECK-NEXT:  sycl::range<3> threaddim = sycl::range<3>(1, 1, 32);
//CHECK-NEXT:  int *karg1, *karg2;
//CHECK-NEXT:  karg1 = sycl::malloc_device<int>(32, q_ct1);
//CHECK-NEXT:  karg2 = sycl::malloc_device<int>(32, q_ct1);
//CHECK-NEXT:  int karg3 = 80;
//CHECK-NEXT:  /*
//CHECK-NEXT:  DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
//CHECK-NEXT:  limit. To get the device limit, query info::device::max_work_group_size.
//CHECK-NEXT:  Adjust the work-group size if needed.
//CHECK-NEXT:  */
//CHECK-NEXT:  q_ct1.parallel_for(sycl::nd_range<3>(griddim * threaddim, threaddim),
//CHECK-NEXT:                     [=](sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:                       testKernelPtr((const int *)karg1, karg2, karg3,
//CHECK-NEXT:                                     item_ct1);
//CHECK-NEXT:                     });
//CHECK-NEXT:}
int main() {
  dim3 griddim = 2;
  dim3 threaddim = 32;
  int *karg1, *karg2;
  cudaMalloc(&karg1, 32 * sizeof(int));
  cudaMalloc(&karg2, 32 * sizeof(int));
  int karg3 = 80;
  testKernelPtr<<<griddim, threaddim>>>((const int *)karg1, karg2, karg3);
}

     //CHECK:#define DEVICE
#define DEVICE __device__

     //CHECK:void DEVICE foo6() { return; }
void DEVICE foo6() {
  return;
}

     //CHECK:struct SharedMemory {
//CHECK-NEXT:  unsigned int *getPointer(uint8_t *dpct_local) {
//CHECK-NEXT:    auto s_uint = (unsigned int *)dpct_local;
//CHECK-NEXT:    return s_uint;
//CHECK-NEXT:  }
//CHECK-NEXT:};
struct SharedMemory
{
  __device__ unsigned int *getPointer()
  {
    extern __shared__ unsigned int s_uint[];
    return s_uint;
  }
};

     //CHECK:typedef struct dpct_type_{{[0-9a-z]+}} {
//CHECK-NEXT:  int SM;
//CHECK-NEXT:  int Cores;
//CHECK-NEXT:} sSMtoCores;
typedef struct
{
  int SM;
  int Cores;
} sSMtoCores;

     //CHECK:#define macro_a (oneapi::mkl::transpose)1
//CHECK-NEXT:void foo7() { oneapi::mkl::transpose a = macro_a; }
#define macro_a (cublasOperation_t)1
void foo7() { cublasOperation_t a = macro_a; }

     //CHECK:void foo8(float *Result) {
//CHECK-NEXT:  for (int i = 0; i < 16; i++) {
//CHECK-NEXT:    if (i % 4 == 0)
//CHECK-NEXT:      Result[i] = log(Result[i]);
//CHECK-NEXT:    if (i % 16 == 0) {
//CHECK-NEXT:      printf("\n");
//CHECK-NEXT:    }
//CHECK-NEXT:    printf("%f ", Result[i]);
//CHECK-NEXT:  }
//CHECK-NEXT:}
void foo8(float *Result) {
  for (int i = 0; i < 16; i++) {
  if (i % 4 == 0) Result[i] = log(Result[i]);
    if (i % 16 == 0) {
      printf("\n");
    }
    printf("%f ", Result[i]);
  }
}

     //CHECK:void foo9(sycl::float2 *Result) {
//CHECK-NEXT:  int a;
//CHECK-NEXT:  int b;
//CHECK-NEXT:}
void foo9(float2 *Result) {
   int a;
 int b;
}