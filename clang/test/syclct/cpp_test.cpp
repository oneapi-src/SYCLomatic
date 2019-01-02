// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck --input-file %T/cpp_test.cc_sycl.cpp --match-full-lines %s

// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <syclct/syclct.hpp>
// CHECK: #include <stdio.h>
// CHECK-NOT:#include <CL/sycl.hpp>
#include <stdio.h>
#include <cuda.h>

#include <cuda_runtime.h>
#include <vector_types.h>

int main()
{
  // CHECK: syclct::sycl_device_info deviceProp;
  cudaDeviceProp deviceProp;
  // CHECK: if (deviceProp.mode() == syclct::compute_mode::prohibited) {
  if (deviceProp.computeMode == cudaComputeModeProhibited) {
    return 0;
  }
  return 0;
}

int printf(const char *format, ...);

// CHECK: void test_00(int err) {
// CHECK-NEXT:   {{ +}}
// CHECK-NEXT: }
void test_00(cudaError_t err) {
  if (err != cudaSuccess) {
    printf("Some error happenned\n");
    exit(1);
  }
}

// CHECK:void test_simple_ifs() {
// CHECK-NEXT:  int err;
// checking for empty lines (with one or more spaces)
// CHECK-NEXT:{{ +}}
// CHECK-NEXT:{{ +}}
// CHECK-NEXT:{{ +}}
// CHECK-NEXT:{{ +}}
// CHECK-NEXT:{{ +}}
// CHECK-NEXT:{{ +}}
// CHECK-NEXT:}
void test_simple_ifs() {
  cudaError_t err;
  if (err != cudaSuccess) {
  }
  if (err) {
  }
  if (err != 0) {
  }
  if (0 != err) {
  }
  if (cudaSuccess != err) {
  }
  if (err != cudaSuccess) {
  }
}

// CHECK:const char *switch_test(int error)
// CHECK-NEXT:{
// CHECK-NEXT:    switch (error)
// CHECK-NEXT:    {
// CHECK-NEXT:        case 0:
// CHECK-NEXT:            return "cudaSuccess";
// CHECK-NEXT:        case 1:
// CHECK-NEXT:            return "cudaErrorMissingConfiguration";
// CHECK-NEXT:        case 2:
// CHECK-NEXT:            return "cudaErrorMemoryAllocation";
// CHECK-NEXT:        case 3:
// CHECK-NEXT:            return "cudaErrorInitializationError";
// CHECK-NEXT:        case 4:
// CHECK-NEXT:            return "cudaErrorLaunchFailure";
// CHECK-NEXT:        case 5:
// CHECK-NEXT:            return "cudaErrorPriorLaunchFailure";
// CHECK-NEXT:        case 6:
// CHECK-NEXT:            return "cudaErrorLaunchTimeout";
// CHECK-NEXT:        case 7:
// CHECK-NEXT:            return "cudaErrorLaunchOutOfResources";
// CHECK-NEXT:        case 8:
// CHECK-NEXT:            return "cudaErrorInvalidDeviceFunction";
// CHECK-NEXT:        case 9:
// CHECK-NEXT:            return "cudaErrorInvalidConfiguration";
// CHECK-NEXT:        case 10:
// CHECK-NEXT:            return "cudaErrorInvalidDevice";
// CHECK-NEXT:        case 11:
// CHECK-NEXT:            return "cudaErrorInvalidValue";
// CHECK-NEXT:        case 12:
// CHECK-NEXT:            return "cudaErrorInvalidPitchValue";
// CHECK-NEXT:        case 13:
// CHECK-NEXT:            return "cudaErrorInvalidSymbol";
// CHECK-NEXT:    }
// CHECK-NEXT:    return 0;
// CHECK-NEXT:}
const char *switch_test(cudaError_t error)
{
    switch (error)
    {
        case cudaSuccess:
            return "cudaSuccess";
        case cudaErrorMissingConfiguration:
            return "cudaErrorMissingConfiguration";
        case cudaErrorMemoryAllocation:
            return "cudaErrorMemoryAllocation";
        case cudaErrorInitializationError:
            return "cudaErrorInitializationError";
        case cudaErrorLaunchFailure:
            return "cudaErrorLaunchFailure";
        case cudaErrorPriorLaunchFailure:
            return "cudaErrorPriorLaunchFailure";
        case cudaErrorLaunchTimeout:
            return "cudaErrorLaunchTimeout";
        case cudaErrorLaunchOutOfResources:
            return "cudaErrorLaunchOutOfResources";
        case cudaErrorInvalidDeviceFunction:
            return "cudaErrorInvalidDeviceFunction";
        case cudaErrorInvalidConfiguration:
            return "cudaErrorInvalidConfiguration";
        case cudaErrorInvalidDevice:
            return "cudaErrorInvalidDevice";
        case cudaErrorInvalidValue:
            return "cudaErrorInvalidValue";
        case cudaErrorInvalidPitchValue:
            return "cudaErrorInvalidPitchValue";
        case cudaErrorInvalidSymbol:
            return "cudaErrorInvalidSymbol";
    }
    return 0;
}

// CHECK: void test_00();
__device__ void test_00();



__global__ void testKernelPtr(const int *L, const int *M, int N) {
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
}

__global__ void testKernel(int L, int M, int N) {
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
}

int kernel_test() {
  dim3 griddim = 2;
  dim3 threaddim = 32;
  void *karg1 = 0;
  const int *karg2 = 0;
  int karg3 = 80;
  // CHECK:  {
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

  // CHECK:  {
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

  // CHECK:  {
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

  // CHECK:  {
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
}
