// RUN: syclct -keep-original-code -out-root %T %s -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck --input-file %T/kernel-call-origcode-embedded.sycl.cpp --match-full-lines %s

#include <iostream>
// includes CUDA
// CHECK:  /*#include <cuda_runtime.h>*/
#include <cuda_runtime.h>

// CHECK:   /*__global__ void testKernelPtr(const int *L, const int *M, int N) {*/
// CHECK-NEXT: void testKernelPtr(const int *L, const int *M, int N, cl::sycl::nd_item<3> [[ITEMNAME:item_[a-f0-9]+]]) {
__global__ void testKernelPtr(const int *L, const int *M, int N) {

  // CHECK: /*  int gtid = blockIdx.x  * blockDim.x */
  // CHECK-NEXT:   int gtid = item_{{[a-f0-9]+}}.get_group(0) /*comments*/ * item_{{[a-f0-9]+}}.get_local_range().get(0) /*comments
  // CHECK-NEXT:  comments*/
  // CHECK-NEXT: /*  + threadIdx.x;*/
  // CHECK-NEXT:  + item_{{[a-f0-9]+}}.get_local_id(0);
  int gtid = blockIdx.x /*comments*/ * blockDim.x /*comments
  comments*/
  + threadIdx.x;
}

// CHECK:     /*__global__ void testKernel(int L, int M, int N) {*/
// CHECK-NEXT: void testKernel(int L, int M, int N, cl::sycl::nd_item<3> [[ITEMNAME:item_[a-f0-9]+]]) {
__global__ void testKernel(int L, int M, int N) {
  // CHECK:      /*  int gtid = blockIdx.x*/
  // CHECK-NEXT:  int gtid = item_{{[a-f0-9]+}}.get_group(0)
  // CHECK-NEXT: /*             * blockDim.x*/
  // CHECK-NEXT:                * item_{{[a-f0-9]+}}.get_local_range().get(0)
  // CHECK-NEXT: /*             + threadIdx.x;*/
  // CHECK-NEXT:                + item_{{[a-f0-9]+}}.get_local_id(0);
  int gtid = blockIdx.x
             * blockDim.x
             + threadIdx.x;
}

// Error handling macro

// CHECK: #define CUDA_CHECK(call) \
// CHECK-NEXT:  /*    if((call) != cudaSuccess) { \*/ \
// CHECK-NEXT:      if((call) != 0) { \
// CHECK-NEXT:  /*        cudaError_t err = cudaGetLastError(); \*/ \
// CHECK-NEXT:          int err = 0; \
// CHECK-NEXT:          std::cout << "CUDA error calling \""#call"\", code is " << err << std::endl; \
// CHECK-NEXT:          exit(err); }
#define CUDA_CHECK(call) \
    if((call) != cudaSuccess) { \
        cudaError_t err = cudaGetLastError(); \
        std::cout << "CUDA error calling \""#call"\", code is " << err << std::endl; \
        exit(err); }

#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )
template< typename T >
void check(T result, char const *const func, const char *const file, int const line)
{}

int main() {
  // CHECK:  /*  dim3 griddim = 2;*/
  // CHECK-NEXT:  cl::sycl::range<3> griddim = cl::sycl::range<3>(2, 1, 1);
  dim3 griddim = 2;

  // CHECK:  /*  dim3 threaddim = 32;*/
  // CHECK-NEXT:   cl::sycl::range<3> threaddim = cl::sycl::range<3>(32, 1, 1);
  dim3 threaddim = 32;

  void *karg1 = 0;
  const int *karg2 = 0;
  int karg3 = 80;
  // CHECK:  /*  testKernelPtr<<<griddim, threaddim>>>((const int *)karg1, karg2, karg3);*/
  // CHECK-NEXT:  {
  // CHECK-NEXT:    std::pair<syclct::buffer_t, size_t> karg1_buf = syclct::get_buffer_and_offset(karg1);
  // CHECK-NEXT:    size_t karg1_offset = karg1_buf.second;
  // CHECK-NEXT:    std::pair<syclct::buffer_t, size_t> karg2_buf = syclct::get_buffer_and_offset(karg2);
  // CHECK-NEXT:    size_t karg2_offset = karg2_buf.second;
  // CHECK-NEXT:    syclct::get_default_queue().submit(
  // CHECK-NEXT:      [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:        auto karg1_acc = karg1_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:        auto karg2_acc = karg2_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:        cgh.parallel_for<syclct_kernel_name<class testKernelPtr_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:          cl::sycl::nd_range<3>((griddim * threaddim), threaddim),
  // CHECK-NEXT:          [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK-NEXT:            void *karg1 = (void*)(&karg1_acc[0] + karg1_offset);
  // CHECK-NEXT:            const int *karg2 = (const int*)(&karg2_acc[0] + karg2_offset);
  // CHECK-NEXT:            testKernelPtr((const int *)karg1, karg2, karg3, [[ITEM]]);
  // CHECK-NEXT:          });
  // CHECK-NEXT:      });
  // CHECK-NEXT:  };
  testKernelPtr<<<griddim, threaddim>>>((const int *)karg1, karg2, karg3);

  // CHECK: /*  testKernel<<<10, intvar>>>(karg1int, karg2int, karg3int);*/
  // CHECK-NEXT:  {
  // CHECK-NEXT:    syclct::get_default_queue().submit(
  // CHECK-NEXT:      [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:        cgh.parallel_for<syclct_kernel_name<class testKernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:          cl::sycl::nd_range<3>((cl::sycl::range<3>(10, 1, 1) * cl::sycl::range<3>(intvar, 1, 1)), cl::sycl::range<3>(intvar, 1, 1)),
  // CHECK-NEXT:          [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK-NEXT:            testKernel(karg1int, karg2int, karg3int, [[ITEM]]);
  // CHECK-NEXT:          });
  // CHECK-NEXT:      });
  // CHECK-NEXT:  };
  int karg1int = 1;
  int karg2int = 2;
  int karg3int = 3;
  int intvar = 20;
  testKernel<<<10, intvar>>>(karg1int, karg2int, karg3int);

  // CHECK: /*  testKernel<<<dim3(1), dim3(1, 2)>>>(karg1int, karg2int, karg3int);*/
  // CHECK-NEXT:  {
  // CHECK-NEXT:    syclct::get_default_queue().submit(
  // CHECK-NEXT:      [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:        cgh.parallel_for<syclct_kernel_name<class testKernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:          cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 2, 1)), cl::sycl::range<3>(1, 2, 1)),
  // CHECK-NEXT:          [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK-NEXT:            testKernel(karg1int, karg2int, karg3int, [[ITEM]]);
  // CHECK-NEXT:          });
  // CHECK-NEXT:      });
  // CHECK-NEXT:  };
  testKernel<<<dim3(1), dim3(1, 2)>>>(karg1int, karg2int, karg3int);

  // CHECK: /*  testKernel<<<dim3(1, 2), dim3(1, 2, 3)>>>(karg1int, karg2int, karg3int);*/
  // CHECK-NEXT:  {
  // CHECK-NEXT:    syclct::get_default_queue().submit(
  // CHECK-NEXT:      [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:        cgh.parallel_for<syclct_kernel_name<class testKernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:          cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 2, 1) * cl::sycl::range<3>(1, 2, 3)), cl::sycl::range<3>(1, 2, 3)),
  // CHECK-NEXT:          [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK-NEXT:            testKernel(karg1int, karg2int, karg3int, [[ITEM]]);
  // CHECK-NEXT:          });
  // CHECK-NEXT:      });
  // CHECK-NEXT:  };
  testKernel<<<dim3(1, 2), dim3(1, 2, 3)>>>(karg1int, karg2int, karg3int);

  // CHECK: /*  testKernel<<<griddim.x, griddim.y + 2>>>(karg1int, karg2int, karg3int);*/
  // CHECK-NEXT:  {
  // CHECK-NEXT:    syclct::get_default_queue().submit(
  // CHECK-NEXT:      [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:      cgh.parallel_for<syclct_kernel_name<class testKernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:        cl::sycl::nd_range<3>((cl::sycl::range<3>(griddim[0], 1, 1) * cl::sycl::range<3>(griddim[1] + 2, 1, 1)), cl::sycl::range<3>(griddim[1] + 2, 1, 1)),
  // CHECK-NEXT:        [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK-NEXT:        testKernel(karg1int, karg2int, karg3int, [[ITEM]]);
  // CHECK-NEXT:      });
  // CHECK-NEXT:    });
  // CHECK-NEXT:  };
  testKernel<<<griddim.x, griddim.y + 2>>>(karg1int, karg2int, karg3int);

  float *deviceOutputData = NULL;

  // CHECK: /*  CUDA_CHECK(cudaMalloc((void **)&deviceOutputData, 10 * sizeof(float)));*/
  // CHECK-NEXT: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Translated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  CUDA_CHECK(cudaMalloc((void **)&deviceOutputData, 10 * sizeof(float)));

  // copy result from device to host
  float *h_odata = NULL;
  float *d_odata = NULL;
  // CHECK: /*  checkCudaErrors(cudaMemcpy(h_odata, d_odata, sizeof(float) * 4, cudaMemcpyDeviceToHost));*/
  // CHECK-NEXT:/*
  // CHECK-NEXT:SYCLCT1003:{{[0-9]+}}: Translated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT:*/
  checkCudaErrors(cudaMemcpy(h_odata, d_odata, sizeof(float) * 4, cudaMemcpyDeviceToHost));
}
