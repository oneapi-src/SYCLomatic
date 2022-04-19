// UNSUPPORTED: cuda-8.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6
// UNSUPPORTED: v8.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6
// RUN: c2s --format-range=none -out-root %T/thrust-for-h2o4gpu-specific-case %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/thrust-for-h2o4gpu-specific-case/thrust-for-h2o4gpu-specific-case.dp.cpp --match-full-lines %s


// CHECK: #include <oneapi/dpl/execution>
// CHECK-NEXT: #include <oneapi/dpl/algorithm>
// CHECK-NEXT: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <c2s/c2s.hpp>
// CHECK-NEXT: #include <c2s/dpl_utils.hpp>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

//CHECK: template <typename T>
//CHECK-NEXT:void foo_cpy(c2s::device_vector<T, sycl::buffer_allocator<T>> &Do, c2s::device_vector<T, sycl::buffer_allocator<T>> &Di) {
//CHECK-NEXT: return;
//CHECK-NEXT:}
template <typename T>
void foo_cpy(thrust::device_vector<T, thrust::device_malloc_allocator<T>> &Do, thrust::device_vector<T, thrust::device_malloc_allocator<T>> &Di) {
 return;
}

//CHECK: void foo_1() {
//CHECK-NEXT: c2s::device_vector<int, sycl::buffer_allocator<int>> **tt=NULL;
//CHECK-NEXT: c2s::device_vector<int, sycl::buffer_allocator<int>> *dd[10];
//CHECK-NEXT: foo_cpy(*tt[0], *dd[0]);
//CHECK-NEXT:}
void foo_1() {
  thrust::device_vector<int, thrust::device_malloc_allocator<int>> **tt=NULL;
  thrust::device_vector<int, thrust::device_malloc_allocator<int>> *dd[10];
  foo_cpy(*tt[0], *dd[0]);
}

