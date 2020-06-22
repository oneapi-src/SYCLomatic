// RUN: cat %s > %T/checkFormatAll.cu
// RUN: cd %T
// RUN: dpct -out-root %T checkFormatAll.cu --cuda-include-path="%cuda-path/include" --format-range=all -- -std=c++14  -x cuda --cuda-host-only
// RUN: FileCheck -strict-whitespace checkFormatAll.cu --match-full-lines --input-file %T/checkFormatAll.dp.cpp

#include <cuda_runtime.h>
#include <cassert>
#include "cublas_v2.h"

     //CHECK:void testDevice(const int *K) { int t = K[0]; }
__device__ void testDevice(const int *K) {
  int t = K[0];
}

     //CHECK:void testDevice1(const int *K) { int t = K[0]; }
__device__ void testDevice1(const int *K) { int t = K[0]; }

     //CHECK:void testKernelPtr(const int *L, const int *M, int N,
//CHECK-NEXT:                   sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:  testDevice(L);
//CHECK-NEXT:  int gtid = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
//CHECK-NEXT:             item_ct1.get_local_id(2);
//CHECK-NEXT:}
__global__ void testKernelPtr(const int *L, const int *M, int N) {
  testDevice(L);
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
}


     //CHECK:int main() {
//CHECK-NEXT:  dpct::device_ext &dev_ct1 = dpct::get_current_device();
//CHECK-NEXT:  sycl::queue &q_ct1 = dev_ct1.default_queue();
//CHECK-NEXT:  sycl::range<3> griddim = sycl::range<3>(2, 1, 1);
//CHECK-NEXT:  sycl::range<3> threaddim = sycl::range<3>(32, 1, 1);
//CHECK-NEXT:  int *karg1, *karg2;
//CHECK-NEXT:  karg1 = sycl::malloc_device<int>(32, q_ct1);
//CHECK-NEXT:  karg2 = sycl::malloc_device<int>(32, q_ct1);
//CHECK-NEXT:  int karg3 = 80;
//CHECK-NEXT:  q_ct1.submit([&](sycl::handler &cgh) {
//CHECK-NEXT:    auto dpct_global_range = griddim * threaddim;
//CHECK-EMPTY:
//CHECK-NEXT:    cgh.parallel_for(
//CHECK-NEXT:        sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
//CHECK-NEXT:                                         dpct_global_range.get(1),
//CHECK-NEXT:                                         dpct_global_range.get(0)),
//CHECK-NEXT:                          sycl::range<3>(threaddim.get(2), threaddim.get(1),
//CHECK-NEXT:                                         threaddim.get(0))),
//CHECK-NEXT:        [=](sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:          testKernelPtr((const int *)karg1, karg2, karg3, item_ct1);
//CHECK-NEXT:        });
//CHECK-NEXT:  });
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

     //CHECK:#define macro_a (mkl::transpose)1
//CHECK-NEXT:void foo7(){ mkl::transpose a = macro_a; }
#define macro_a (cublasOperation_t)1
void foo7(){ cublasOperation_t a = macro_a; }