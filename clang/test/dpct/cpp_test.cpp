// RUN: dpct --format-range=none --usm-level=none -out-root %T/cpp_test %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cpp_test/cpp_test.cpp.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/cpp_test/cpp_test.cpp.dp.cpp -o %T/cpp_test/cpp_test.o.dp.o %}

// CHECK: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK: #include <stdio.h>
// CHECK-NOT:#include <sycl/sycl.hpp>
#include <stdio.h>
#include <cuda.h>

#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector>
int main()
{
  // CHECK: dpct::device_info deviceProp;
  cudaDeviceProp deviceProp;
  // CHECK: /*
  // CHECK-NEXT: DPCT1035:{{[0-9]+}}: All SYCL devices can be used by the host to submit tasks. You may need to adjust this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (true) {
  if (deviceProp.computeMode == cudaComputeModeDefault) {
    return 0;
  }
  // CHECK: /*
  // CHECK-NEXT: DPCT1035:{{[0-9]+}}: All SYCL devices can be used by the host to submit tasks. You may need to adjust this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (true) {
  if (cudaComputeModeDefault == deviceProp.computeMode) {
    return 0;
  }
  // CHECK: /*
  // CHECK-NEXT: DPCT1035:{{[0-9]+}}: All SYCL devices can be used by the host to submit tasks. You may need to adjust this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (false) {
  if (deviceProp.computeMode != cudaComputeModeDefault) {
    return 0;
  }
  // CHECK: /*
  // CHECK-NEXT: DPCT1035:{{[0-9]+}}: All SYCL devices can be used by the host to submit tasks. You may need to adjust this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (false) {
  if (cudaComputeModeDefault != deviceProp.computeMode) {
    return 0;
  }
  // CHECK: /*
  // CHECK-NEXT: DPCT1035:{{[0-9]+}}: All SYCL devices can be used by the host to submit tasks. You may need to adjust this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (false) {
  if (deviceProp.computeMode == cudaComputeModeExclusive) {
    return 0;
  }
  // CHECK: /*
  // CHECK-NEXT: DPCT1035:{{[0-9]+}}: All SYCL devices can be used by the host to submit tasks. You may need to adjust this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (true) {
  if (deviceProp.computeMode != cudaComputeModeExclusive) {
    return 0;
  }
  // CHECK: /*
  // CHECK-NEXT: DPCT1035:{{[0-9]+}}: All SYCL devices can be used by the host to submit tasks. You may need to adjust this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (false) {
  if (deviceProp.computeMode == cudaComputeModeProhibited) {
    return 0;
  }
  // CHECK: /*
  // CHECK-NEXT: DPCT1035:{{[0-9]+}}: All SYCL devices can be used by the host to submit tasks. You may need to adjust this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (true) {
  if (deviceProp.computeMode != cudaComputeModeProhibited) {
    return 0;
  }
  // CHECK: /*
  // CHECK-NEXT: DPCT1035:{{[0-9]+}}: All SYCL devices can be used by the host to submit tasks. You may need to adjust this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (false) {
  if (deviceProp.computeMode == cudaComputeModeExclusiveProcess) {
    return 0;
  }
  // CHECK: /*
  // CHECK-NEXT: DPCT1035:{{[0-9]+}}: All SYCL devices can be used by the host to submit tasks. You may need to adjust this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (true) {
  if (deviceProp.computeMode != cudaComputeModeExclusiveProcess) {
    return 0;
  }
  std::vector<cudaDeviceProp> deviceProps;
  int i;
  // CHECK: /*
  // CHECK-NEXT: DPCT1035:{{[0-9]+}}: All SYCL devices can be used by the host to submit tasks. You may need to adjust this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (true) {
  if (deviceProps[i].computeMode != cudaComputeModeExclusiveProcess) {
    return 0;
  }

  int cm = 0;
  // CHECK: /*
  // CHECK-NEXT: DPCT1035:{{[0-9]+}}: All SYCL devices can be used by the host to submit tasks. You may need to adjust this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (cm == 1) {
  if (cm == cudaComputeModeDefault) {
    return 0;
  }
  // CHECK: /*
  // CHECK-NEXT: DPCT1035:{{[0-9]+}}: All SYCL devices can be used by the host to submit tasks. You may need to adjust this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (cm == 0) {
  if (cm == cudaComputeModeExclusive) {
    return 0;
  }
  // CHECK: /*
  // CHECK-NEXT: DPCT1035:{{[0-9]+}}: All SYCL devices can be used by the host to submit tasks. You may need to adjust this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (cm == 0) {
  if (cm == cudaComputeModeProhibited) {
    return 0;
  }
  // CHECK: /*
  // CHECK-NEXT: DPCT1035:{{[0-9]+}}: All SYCL devices can be used by the host to submit tasks. You may need to adjust this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (cm == 0) {
  if (cm == cudaComputeModeExclusiveProcess) {
    return 0;
  }
  return 0;
}

int printf(const char *format, ...);

// CHECK: void test_00(dpct::err0 err) {
// CHECK-NEXT:   {{ +}}
// CHECK-NEXT: }
void test_00(cudaError_t err) {
  if (err != cudaSuccess) {
    printf("Some error happenned\n");
    exit(1);
  }
}

// CHECK:void test_simple_ifs() {
// CHECK-NEXT:  dpct::err0 err;
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

// CHECK:const char *switch_test(dpct::err0 error)
// CHECK-NEXT:{
// CHECK-NEXT:    switch (error)
// CHECK-NEXT:    {
// CHECK-NEXT:        case 0:
// CHECK-NEXT:            return "cudaSuccess";
// CHECK-NEXT:        case {{[0-9]+}}:
// CHECK-NEXT:            return "cudaErrorMissingConfiguration";
// CHECK-NEXT:        case 2:
// CHECK-NEXT:            return "cudaErrorMemoryAllocation";
// CHECK-NEXT:        case 3:
// CHECK-NEXT:            return "cudaErrorInitializationError";
// CHECK-NEXT:        case {{[0-9]+}}:
// CHECK-NEXT:            return "cudaErrorLaunchFailure";
// CHECK-NEXT:        case {{[0-9]+}}:
// CHECK-NEXT:            return "cudaErrorPriorLaunchFailure";
// CHECK-NEXT:        case {{[0-9]+}}:
// CHECK-NEXT:            return "cudaErrorLaunchTimeout";
// CHECK-NEXT:        case {{[0-9]+}}:
// CHECK-NEXT:            return "cudaErrorLaunchOutOfResources";
// CHECK-NEXT:        case {{[0-9]+}}:
// CHECK-NEXT:            return "cudaErrorInvalidDeviceFunction";
// CHECK-NEXT:        case dpct::default_error:
// CHECK-NEXT:            return "cudaErrorInvalidConfiguration";
// CHECK-NEXT:        case {{[0-9]+}}:
// CHECK-NEXT:            return "cudaErrorInvalidDevice";
// CHECK-NEXT:        case {{[0-9]+}}:
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
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.out_of_order_queue();
  dim3 griddim = 2;
  dim3 threaddim = 32;
  void *karg1 = 0;
  const int *karg2 = 0;
  int karg3 = 80;
  // CHECK: q_ct1.submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper karg1_acc_ct0((const int *)karg1, cgh);
  // CHECK-NEXT:     dpct::access_wrapper karg2_acc_ct1(karg2, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class testKernelPtr_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(griddim * threaddim, threaddim),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         testKernelPtr(karg1_acc_ct0.get_raw_pointer(), karg2_acc_ct1.get_raw_pointer(), karg3, item_ct1);
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  testKernelPtr<<<griddim, threaddim>>>((const int *)karg1, karg2, karg3);

  int karg1int = 1;
  int karg2int = 2;
  int karg3int = 3;
  int intvar = 20;
  // CHECK:   q_ct1.parallel_for<dpct_kernel_name<class testKernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 10) * sycl::range<3>(1, 1, intvar), sycl::range<3>(1, 1, intvar)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           testKernel(karg1int, karg2int, karg3int, item_ct1);
  // CHECK-NEXT:         });
  testKernel<<<10, intvar>>>(karg1int, karg2int, karg3int);

  // CHECK:   q_ct1.parallel_for<dpct_kernel_name<class testKernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 2, 1), sycl::range<3>(1, 2, 1)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           testKernel(karg1int, karg2int, karg3int, item_ct1);
  // CHECK-NEXT:         });
  testKernel<<<dim3(1), dim3(1, 2)>>>(karg1int, karg2int, karg3int);

  // CHECK:   q_ct1.parallel_for<dpct_kernel_name<class testKernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 2, 1) * sycl::range<3>(3, 2, 1), sycl::range<3>(3, 2, 1)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           testKernel(karg1int, karg2int, karg3int, item_ct1);
  // CHECK-NEXT:         });
  testKernel<<<dim3(1, 2), dim3(1, 2, 3)>>>(karg1int, karg2int, karg3int);
}
