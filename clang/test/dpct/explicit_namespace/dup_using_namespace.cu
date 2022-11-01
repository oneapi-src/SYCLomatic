// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0
// RUN: dpct --format-range=none -out-root %T/dup_using_namespace %s --cuda-include-path="%cuda-path/include" --use-explicit-namespace=sycl -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/dup_using_namespace/dup_using_namespace.dp.cpp --match-full-lines %s
// CHECK: #include <oneapi/dpl/execution>
// CHECK-NEXT: #include <oneapi/dpl/algorithm>
// CHECK-NEXT: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: using namespace dpct;
// CHECK-NEXT: #include <complex>
// CHECK-NEXT: #include <dpct/dpl_utils.hpp>
// CHECK-EMPTY:
#include <thrust/complex.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <cuda_runtime.h>

// CHECK: void kernel(std::complex<double> *det) {}
__global__ void kernel(thrust::complex<double> *det) {}

int main() {
  thrust::device_ptr<thrust::complex<double>> d_ptr = thrust::device_malloc<thrust::complex<double>>(1);
  kernel<<<1,256>>>(thrust::raw_pointer_cast(d_ptr));
  std::complex<double> res;
  cudaMemcpy(std::addressof(res), thrust::raw_pointer_cast(d_ptr), sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
  thrust::device_free(d_ptr);
}

