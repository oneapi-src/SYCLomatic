// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2
// RUN: dpct --format-range=none -out-root %T/dim3 %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/dim3/dim3.dp.cpp --match-full-lines %s

#include <cuda.h>

int main() {
  // CHECK: /*
  // CHECK-NEXT: DPCT1082:{{[0-9]+}}: Migration of cudaKernelNodeParams type is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT: cudaKernelNodeParams kernelNodeParam0 = {};
  cudaKernelNodeParams kernelNodeParam0 = {};
  // CHECK: /*
  // CHECK-NEXT: DPCT1082:{{[0-9]+}}: Migration of cudaKernelNodeParams type is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT: cudaKernelNodeParams kernelNodeParam1 = {0};
  cudaKernelNodeParams kernelNodeParam1 = {0};
  // CHECK: /*
  // CHECK-NEXT: DPCT1082:{{[0-9]+}}: Migration of cudaKernelNodeParams type is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT: cudaKernelNodeParams kernelNodeParam2 = {0, sycl::range<3>(1, 1, 0)};
  cudaKernelNodeParams kernelNodeParam2 = {0, 0};
  // CHECK: /*
  // CHECK-NEXT: DPCT1082:{{[0-9]+}}: Migration of cudaKernelNodeParams type is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT: cudaKernelNodeParams kernelNodeParam3 = {0, sycl::range<3>(1, 1, 0), sycl::range<3>(1, 1, 0)};
  cudaKernelNodeParams kernelNodeParam3 = {0, 0, 0};

  // CHECK: /*
  // CHECK-NEXT: DPCT1082:{{[0-9]+}}: Migration of cudaKernelNodeParams type is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT: cudaKernelNodeParams kernelNodeParam4{};
  cudaKernelNodeParams kernelNodeParam4{};
  // CHECK: /*
  // CHECK-NEXT: DPCT1082:{{[0-9]+}}: Migration of cudaKernelNodeParams type is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT: cudaKernelNodeParams kernelNodeParam5{0};
  cudaKernelNodeParams kernelNodeParam5{0};
  // CHECK: /*
  // CHECK-NEXT: DPCT1082:{{[0-9]+}}: Migration of cudaKernelNodeParams type is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT: cudaKernelNodeParams kernelNodeParam6{0, sycl::range<3>(1, 1, 0)};
  cudaKernelNodeParams kernelNodeParam6{0, 0};
  // CHECK: /*
  // CHECK-NEXT: DPCT1082:{{[0-9]+}}: Migration of cudaKernelNodeParams type is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT: cudaKernelNodeParams kernelNodeParam7{0, sycl::range<3>(1, 1, 0), sycl::range<3>(1, 1, 0)};
  cudaKernelNodeParams kernelNodeParam7{0, 0, 0};
}
