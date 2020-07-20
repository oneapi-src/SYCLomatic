// UNSUPPORTED: cuda-8.0, cuda-10.1, cuda-10.2, cuda-11.0
// UNSUPPORTED: v8.0, v10.1, v10.2, v11.0
// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/thrust-for-h2o4gpu-specific-case.dp.cpp --match-full-lines %s


// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <dpct/dpstd_utils.hpp>
// CHECK-NEXT: #include <dpstd/execution>
// CHECK-NEXT: #include <dpstd/algorithm>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

//CHECK: template <typename T>
//CHECK-NEXT:void foo_cpy(dpct::device_vector<T, dpct::device_malloc_allocator<T>> &Do, dpct::device_vector<T, dpct::device_malloc_allocator<T>> &Di) {
//CHECK-NEXT: return;
//CHECK-NEXT:}
template <typename T>
void foo_cpy(thrust::device_vector<T, thrust::device_malloc_allocator<T>> &Do, thrust::device_vector<T, thrust::device_malloc_allocator<T>> &Di) {
 return;
}

//CHECK: void foo_1() {
//CHECK-NEXT: dpct::device_vector<int, dpct::device_malloc_allocator<int>> **tt=NULL;
//CHECK-NEXT: dpct::device_vector<int, dpct::device_malloc_allocator<int>> *dd[10];
//CHECK-NEXT: foo_cpy(*tt[0], *dd[0]);
//CHECK-NEXT:}
void foo_1() {
  thrust::device_vector<int, thrust::device_malloc_allocator<int>> **tt=NULL;
  thrust::device_vector<int, thrust::device_malloc_allocator<int>> *dd[10];
  foo_cpy(*tt[0], *dd[0]);
}
