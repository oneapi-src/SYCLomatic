// RUN: dpct --format-range=none --usm-level=none -keep-original-code -out-root %T/kernel-call-origcode-embedded %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/kernel-call-origcode-embedded/kernel-call-origcode-embedded.dp.cpp --match-full-lines %s

#include <iostream>
// includes CUDA
// CHECK:  /* DPCT_ORIG #include <cuda_runtime.h>*/
#include <cuda_runtime.h>

// CHECK:   /* DPCT_ORIG __global__ void testKernelPtr(const int *L, const int *M, int N) {*/
// CHECK-NEXT:void testKernelPtr(const int *L, const int *M, int N, sycl::nd_item<3> [[ITEMNAME:item_ct1]]) {
__global__ void testKernelPtr(const int *L, const int *M, int N) {
  // CHECK: /* DPCT_ORIG   int gtid = blockIdx.x  * blockDim.x */
  // CHECK-NEXT:   int gtid = item_ct1.get_group(2) /*comments*/ * item_ct1.get_local_range(2) /*comments
  // CHECK-NEXT:  comments*/
  // CHECK-NEXT: /* DPCT_ORIG   + threadIdx.x;*/
  // CHECK-NEXT:  + item_ct1.get_local_id(2);
  int gtid = blockIdx.x /*comments*/ * blockDim.x /*comments
  comments*/
             + threadIdx.x;
}

// CHECK:     /* DPCT_ORIG __global__ void testKernel(int L, int M, int N) {*/
// CHECK-NEXT: void testKernel(int L, int M, int N, sycl::nd_item<3> [[ITEMNAME:item_ct1]]) {
__global__ void testKernel(int L, int M, int N) {
  // CHECK:      /* DPCT_ORIG   int gtid = blockIdx.x*/
  // CHECK-NEXT:  int gtid = item_ct1.get_group(2)
  // CHECK-NEXT: /* DPCT_ORIG              * blockDim.x*/
  // CHECK-NEXT:                * item_ct1.get_local_range(2)
  // CHECK-NEXT: /* DPCT_ORIG              + threadIdx.x;*/
  // CHECK-NEXT:                + item_ct1.get_local_id(2);
  int gtid = blockIdx.x
             * blockDim.x
             + threadIdx.x;
}

// Error handling macro

// CHECK: #define MY_CHECKER(CALL) \
// CHECK-NEXT: /* DPCT_ORIG  if ((CALL) != cudaSuccess) { \*/ \
// CHECK-NEXT:   if ((CALL) != 0) { \
// CHECK-NEXT:     exit(-1); \
// CHECK-NEXT:   }
#define MY_CHECKER(CALL) \
  if ((CALL) != cudaSuccess) { \
    exit(-1); \
  }

#define MY_ERROR_CHECKER(CALL) my_error_checker((CALL), #CALL)
template <typename T>
void my_error_checker(T ReturnValue, char const *const FuncName) {}

int main() {
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
  // CHECK:  /* DPCT_ORIG   dim3 griddim = 2;*/
  // CHECK-NEXT:  sycl::range<3> griddim = sycl::range<3>(1, 1, 2);
  dim3 griddim = 2;

  // CHECK:  /* DPCT_ORIG   dim3 threaddim = 32;*/
  // CHECK-NEXT:   sycl::range<3> threaddim = sycl::range<3>(1, 1, 32);
  dim3 threaddim = 32;

  void *karg1 = 0;
  const int *karg2 = 0;
  int karg3 = 80;
  // CHECK:  /* DPCT_ORIG   testKernelPtr<<<griddim, threaddim>>>((const int *)karg1,
  // CHECK-NEXT:  karg2, karg3);*/
  // CHECK: /*
  // CHECK-NEXT: DPCT1049:{{[0-9]+}}: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  // CHECK-NEXT: */
  // CHECK-NEXT: q_ct1.submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<const int *> karg1_acc_ct0((const int *)karg1, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<const int *> karg2_acc_ct1(karg2, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class testKernelPtr_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(griddim * threaddim, threaddim),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         testKernelPtr(karg1_acc_ct0.get_raw_pointer(), karg2_acc_ct1.get_raw_pointer(), karg3, item_ct1);
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  testKernelPtr<<<griddim, threaddim>>>((const int *)karg1,
                                        karg2, karg3);


  int karg1int = 1;
  int karg2int = 2;
  int karg3int = 3;
  int intvar = 20;
  // CHECK: /* DPCT_ORIG   testKernel<<<10, intvar>>>(karg1int, karg2int, 
  // CHECK:  karg3int);*/
  // CHECK:   /*
  // CHECK-NEXT:   DPCT1049:{{[0-9]+}}: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   q_ct1.parallel_for<dpct_kernel_name<class testKernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 10) * sycl::range<3>(1, 1, intvar), sycl::range<3>(1, 1, intvar)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           testKernel(karg1int, karg2int, karg3int, item_ct1);
  // CHECK-NEXT:         });
  testKernel<<<10, intvar>>>(karg1int, karg2int, // comments
                             // comments.
                             karg3int);

  // CHECK: /* DPCT_ORIG   testKernel<<<dim3(1), dim3(1, 2)>>>(karg1int,
  // CHECK:  karg2int,
  // CHECK:  karg3int);*/
  // CHECK-NEXT:   q_ct1.parallel_for<dpct_kernel_name<class testKernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 2, 1), sycl::range<3>(1, 2, 1)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           testKernel(karg1int, karg2int, karg3int, item_ct1);
  // CHECK-NEXT:         });
  testKernel<<<dim3(1), dim3(1, 2)>>>(karg1int,
                                      /* comments */
                                      karg2int, // comments
                                      /*
                                      comments
                                      */
                                      karg3int);

  // CHECK: /* DPCT_ORIG   testKernel<<<dim3(1, 2), dim3(1, 2, 3)>>>(karg1int,
  // CHECK-NEXT:  karg2int, karg3int); */
  // CHECK-NEXT:   q_ct1.parallel_for<dpct_kernel_name<class testKernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 2, 1) * sycl::range<3>(3, 2, 1), sycl::range<3>(3, 2, 1)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           testKernel(karg1int, karg2int, karg3int, item_ct1);
  // CHECK-NEXT:         }); // comments
  testKernel<<<dim3(1, 2), dim3(1, 2, 3)>>>(karg1int,
	  karg2int, /* comments */karg3int/* comments */); // comments

  // CHECK: /* DPCT_ORIG   testKernel<<<griddim.x, griddim.y + 2>>>(karg1int, karg2int, karg3int);*/
  // CHECK:   /*
  // CHECK-NEXT:   DPCT1049:{{[0-9]+}}: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   q_ct1.parallel_for<dpct_kernel_name<class testKernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, griddim[2]) * sycl::range<3>(1, 1, griddim[1] + 2), sycl::range<3>(1, 1, griddim[1] + 2)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           testKernel(karg1int, karg2int, karg3int, item_ct1);
  // CHECK-NEXT:         });
  testKernel<<<griddim.x, griddim.y + 2>>>(karg1int, karg2int, karg3int);

  float *deviceOutputData = NULL;

  // CHECK: /* DPCT_ORIG   MY_CHECKER(cudaMalloc((void **)&deviceOutputData, 10 * sizeof(float)));*/
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  MY_CHECKER(cudaMalloc((void **)&deviceOutputData, 10 * sizeof(float)));

  // copy result from device to host
  float *h_odata = NULL;
  float *d_odata = NULL;
  // CHECK: /* DPCT_ORIG   MY_ERROR_CHECKER(cudaMemcpy(h_odata, d_odata, sizeof(float) * 4, cudaMemcpyDeviceToHost));*/
  // CHECK-NEXT:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  MY_ERROR_CHECKER(cudaMemcpy(h_odata, d_odata, sizeof(float) * 4, cudaMemcpyDeviceToHost));

  // CHECK: /*
  // CHECK-NEXT:DPCT1007:{{[0-9]+}}: Migration of cudaThreadGetCacheConfig is not supported.
  // CHECK-NEXT:*/
  cudaThreadGetCacheConfig(NULL);

  // CHECK: /* DPCT_ORIG   cudaThreadGetCacheConfig(NULL);cudaMalloc((void **)&deviceOutputData, 10 * sizeof(float));*/
  // CHECK-NEXT: /*
  // CHECK-NEXT:  DPCT1007:{{[0-9]+}}: Migration of cudaThreadGetCacheConfig is not supported.
  // CHECK-NEXT: */
  cudaThreadGetCacheConfig(NULL);cudaMalloc((void **)&deviceOutputData, 10 * sizeof(float));

  // CHECK: /* DPCT_ORIG   cudaEventCreate(NULL);MY_ERROR_CHECKER(cudaMemcpy(h_odata, d_odata, sizeof(float) * 4, cudaMemcpyDeviceToHost));MY_ERROR_CHECKER(cudaMemcpy(h_odata, d_odata, sizeof(float) * 4, cudaMemcpyDeviceToHost));*/
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudaEventCreate was removed because this call is redundant in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT:  /*
  // CHECK-NEXT:  DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:  */
  cudaEventCreate(NULL);MY_ERROR_CHECKER(cudaMemcpy(h_odata, d_odata, sizeof(float) * 4, cudaMemcpyDeviceToHost));MY_ERROR_CHECKER(cudaMemcpy(h_odata, d_odata, sizeof(float) * 4, cudaMemcpyDeviceToHost));
}

// CHECK: /* DPCT_ORIG template <bool storeSum, bool isNP2>
// CHECK-NEXT:__global__ static void foo_2(unsigned int *g_odata,
// CHECK-NEXT:                            const unsigned int *g_idata,
// CHECK-NEXT:                            unsigned int *g_blockSums,
// CHECK-NEXT:                            int n,
// CHECK-NEXT:                            int blockIndex,
// CHECK-NEXT:                            int baseIndex);*/
// CHECK-NEXT:template <bool storeSum, bool isNP2>
// CHECK-NEXT:static void foo_2(unsigned int *g_odata,
// CHECK-NEXT:                            const unsigned int *g_idata,
// CHECK-NEXT:                            unsigned int *g_blockSums,
// CHECK-NEXT:                            int n,
// CHECK-NEXT:                            int blockIndex,
// CHECK-NEXT:                            int baseIndex,
// CHECK-NEXT:                            sycl::nd_item<3> item_ct1,
// CHECK-NEXT:                            uint8_t *dpct_local);
template <bool storeSum, bool isNP2>
__global__ static void foo_2(unsigned int *g_odata,
                            const unsigned int *g_idata,
                            unsigned int *g_blockSums,
                            int n,
                            int blockIndex,
                            int baseIndex);

// CHECK: /* DPCT_ORIG template <bool isNP2>
// CHECK-NEXT:__device__ static void foo_1(unsigned int* g_odata,
// CHECK-NEXT:                          const unsigned int* s_data,
// CHECK-NEXT:                          int n,
// CHECK-NEXT:                          int ai, int bi,
// CHECK-NEXT:                          int mem_ai, int mem_bi,
// CHECK-NEXT:                          int bankOffsetA, int bankOffsetB);*/
// CHECK-NEXT:template <bool isNP2>
// CHECK-NEXT:static void foo_1(unsigned int* g_odata,
// CHECK-NEXT:                          const unsigned int* s_data,
// CHECK-NEXT:                          int n,
// CHECK-NEXT:                          int ai, int bi,
// CHECK-NEXT:                          int mem_ai, int mem_bi,
// CHECK-NEXT:                          int bankOffsetA, int bankOffsetB,
// CHECK-NEXT:                          sycl::nd_item<3> item_ct1);
template <bool isNP2>
__device__ static void foo_1(unsigned int* g_odata,
                          const unsigned int* s_data,
                          int n,
                          int ai, int bi,
                          int mem_ai, int mem_bi,
                          int bankOffsetA, int bankOffsetB);

// CHECK:/* DPCT_ORIG template <bool isNP2>
// CHECK-NEXT:__device__ static void foo_1(unsigned int* g_odata,
// CHECK-NEXT:                              const unsigned int* s_data,
// CHECK-NEXT:                              int n,
// CHECK-NEXT:                              int ai, int bi,
// CHECK-NEXT:                              int mem_ai, int mem_bi,
// CHECK-NEXT:                              int bankOffsetA, int bankOffsetB)*/
// CHECK-NEXT: template <bool isNP2>
// CHECK-NEXT:static void foo_1(unsigned int* g_odata,
// CHECK-NEXT:                              const unsigned int* s_data,
// CHECK-NEXT:                              int n,
// CHECK-NEXT:                              int ai, int bi,
// CHECK-NEXT:                              int mem_ai, int mem_bi,
// CHECK-NEXT:                              int bankOffsetA, int bankOffsetB,
// CHECK-NEXT:                              sycl::nd_item<3> item_ct1)
template <bool isNP2>
__device__ static void foo_1(unsigned int* g_odata,
                              const unsigned int* s_data,
                              int n,
                              int ai, int bi,
                              int mem_ai, int mem_bi,
                              int bankOffsetA, int bankOffsetB)
{
// CHECK: /* DPCT_ORIG     __syncthreads();*/
// CHECK-NEXT:    sycl::group_barrier(item_ct1.get_group());
    __syncthreads();
}

// CHECK:/* DPCT_ORIG template <bool storeSum, bool isNP2>
// CHECK-NEXT:__global__ static void foo_2(unsigned int *g_odata,
// CHECK-NEXT:                        const unsigned int *g_idata,
// CHECK-NEXT:                        unsigned int *g_blockSums,
// CHECK-NEXT:                        int n,
// CHECK-NEXT:                        int blockIndex,
// CHECK-NEXT:                        int baseIndex)*/
// CHECK-NEXT:template <bool storeSum, bool isNP2>
// CHECK-NEXT:static void foo_2(unsigned int *g_odata,
// CHECK-NEXT:                        const unsigned int *g_idata,
// CHECK-NEXT:                        unsigned int *g_blockSums,
// CHECK-NEXT:                        int n,
// CHECK-NEXT:                        int blockIndex,
// CHECK-NEXT:                        int baseIndex,
// CHECK-NEXT:                        sycl::nd_item<3> item_ct1,
// CHECK-NEXT:                        uint8_t *dpct_local)
template <bool storeSum, bool isNP2>
__global__ static void foo_2(unsigned int *g_odata,
                        const unsigned int *g_idata,
                        unsigned int *g_blockSums,
                        int n,
                        int blockIndex,
                        int baseIndex)
{
    int ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB;
    extern __shared__ unsigned int s_data[];
    foo_1<isNP2>(g_odata, s_data, n,
                                 ai, bi, mem_ai, mem_bi,
                                 bankOffsetA, bankOffsetB);
}

// CHECK: /* DPCT_ORIG __global__ static void foo_3(void){*/
// CHECK-NEXT: static void foo_3(){
__global__ static void foo_3(void){
}

int foo_test_1358() {
 foo_2<true, true><<<1, 1>>>(NULL, NULL, NULL, 1, 2, 3);
 return 0;
}

