// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0
// RUN: dpct --format-range=none -out-root %T/thrust-cast %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/thrust-cast/thrust-cast.dp.cpp --match-full-lines %s
// CHECK: #include <oneapi/dpl/execution>
// CHECK-NEXT: #include <oneapi/dpl/algorithm>
// CHECK-NEXT: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <complex>
// CHECK-NEXT: #include <dpct/dpl_utils.hpp>
#include <thrust/complex.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <cuda_runtime.h>

// CHECK: void kernel(std::complex<double> *det) {}
__global__ void kernel(thrust::complex<double> *det) {}

int main() {
// CHECK:  dpct::device_pointer<std::complex<double>> d_ptr = dpct::malloc_device<std::complex<double>>(1);
  thrust::device_ptr<thrust::complex<double>> d_ptr = thrust::device_malloc<thrust::complex<double>>(1);
// CHECK:  q_ct1.submit(
// CHECK-NEXT:    [&](sycl::handler &cgh) {
// CHECK-NEXT:      auto thrust_raw_pointer_cast_d_ptr_ct0 = dpct::get_raw_pointer(d_ptr);
// CHECK-EMPTY:
// CHECK-NEXT:      cgh.parallel_for(
// CHECK-NEXT:        sycl::nd_range<3>(sycl::range<3>(1, 1, 256), sycl::range<3>(1, 1, 256)),
// CHECK-NEXT:        [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:          kernel(thrust_raw_pointer_cast_d_ptr_ct0);
// CHECK-NEXT:        });
// CHECK-NEXT:    });
  kernel<<<1,256>>>(thrust::raw_pointer_cast(d_ptr));
  std::complex<double> res;
// CHECK:  q_ct1.memcpy(std::addressof(res), dpct::get_raw_pointer(d_ptr), sizeof(std::complex<double>)).wait();
  cudaMemcpy(std::addressof(res), thrust::raw_pointer_cast(d_ptr), sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
// CHECK:  dpct::free_device(d_ptr);
  thrust::device_free(d_ptr);
}

