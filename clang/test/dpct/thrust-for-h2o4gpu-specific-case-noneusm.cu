// UNSUPPORTED: cuda-8.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3
// UNSUPPORTED: v8.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3
// RUN: dpct --usm-level=none --format-range=none -out-root %T/thrust-for-h2o4gpu-specific-case-noneusm %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/thrust-for-h2o4gpu-specific-case-noneusm/thrust-for-h2o4gpu-specific-case-noneusm.dp.cpp --match-full-lines %s

// CHECK: #include <oneapi/dpl/execution>
// CHECK-NEXT:#include <oneapi/dpl/algorithm>
// CHECK-NEXT:#define DPCT_USM_LEVEL_NONE
// CHECK-NEXT:#include <sycl/sycl.hpp>
// CHECK-NEXT:#include <dpct/dpct.hpp>
// CHECK-NEXT:#include <dpct/dpl_utils.hpp>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/device_allocator.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/iterator/iterator_traits.h>

// CHECK: /*
// CHECK-NEXT:DPCT1082:{{[0-9]+}}: Migration of thrust::device_malloc_allocator type is not supported.
// CHECK-NEXT:*/
// CHECK-NEXT:template <typename T> void foo_cpy(dpct::device_vector<T, thrust::device_malloc_allocator<T>> &Do, dpct::device_vector<T, thrust::device_malloc_allocator<T>> &Di) {
// CHECK-NEXT: return;
// CHECK-NEXT:}
template <typename T> void foo_cpy(thrust::device_vector<T, thrust::device_malloc_allocator<T>> &Do, thrust::device_vector<T, thrust::device_malloc_allocator<T>> &Di) {
 return;
}


void foo_1() {
// CHECK:  /*
// CHECK-NEXT:  DPCT1082:{{[0-9]+}}: Migration of thrust::device_malloc_allocator type is not supported.
// CHECK-NEXT:  */
// CHECK-NEXT:  dpct::device_vector<int, thrust::device_malloc_allocator<int>> **tt=NULL;
// CHECK-NEXT:  /*
// CHECK-NEXT:  DPCT1082:{{[0-9]+}}: Migration of thrust::device_malloc_allocator type is not supported.
// CHECK-NEXT:  */
// CHECK-NEXT:  dpct::device_vector<int, thrust::device_malloc_allocator<int>> *dd[10];
// CHECK-NEXT:  foo_cpy(*tt[0], *dd[0]);
  thrust::device_vector<int, thrust::device_malloc_allocator<int>> **tt=NULL;
  thrust::device_vector<int, thrust::device_malloc_allocator<int>> *dd[10];
  foo_cpy(*tt[0], *dd[0]);
}

// CHECK:/*
// CHECK-NEXT:DPCT1082:{{[0-9]+}}: Migration of thrust::device_malloc_allocator type is not supported.
// CHECK-NEXT:*/
namespace foo_struct { template <typename T, typename MemorySpace> struct default_memory_allocator : thrust::device_malloc_allocator<T>{};}
