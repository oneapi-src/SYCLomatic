// UNSUPPORTED: cuda-8.0, cuda-12.0, cuda-12.1, cuda-12.2, cuda-12.3, cuda-12.4, cuda-12.5, cuda-12.6
// UNSUPPORTED: v8.0, v12.0, v12.1, v12.2, v12.3, v12.4, v12.5, v12.6
// RUN: dpct --format-range=none --use-syclcompat -out-root %T/allocator_syclcompat %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --match-full-lines --input-file %T/allocator_syclcompat/allocator_syclcompat.dp.cpp %s
// RUN: %if build_lit %{icpx -c -fsycl -DBUILD_TEST %T/allocator_syclcompat/allocator_syclcompat.dp.cpp -o %T/allocator_syclcompat/allocator_syclcompat.dp.o %}

#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/device_allocator.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <thrust/transform.h>
#include <vector>

#define SIZE 4

template<class T>
int foo() {
#ifndef BUILD_TEST
  // CHECK: DPCT1131:{{[0-9]+}}: The migration of "thrust::system::cuda::experimental::pinned_allocator" is not currently supported with SYCLcompat. Please adjust the code manually.
  std::vector<float, thrust::system::cuda::experimental::pinned_allocator<float>> hVec(SIZE);

  // CHECK: DPCT1131:{{[0-9]+}}: The migration of "thrust::system::cuda::experimental::pinned_allocator" is not currently supported with SYCLcompat. Please adjust the code manually.
  std::vector<float, thrust::cuda::experimental::pinned_allocator<float>> hVecCopy = hVec;

  // CHECK: DPCT1131:{{[0-9]+}}: The migration of "thrust::device_allocator" is not currently supported with SYCLcompat. Please adjust the code manually.
  thrust::device_vector<T, thrust::device_allocator<T>> dvec;
#endif

  return 0;
}

template int foo<int>();
