// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2
// RUN: dpct --format-range=none -out-root %T/thrust_misc %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/thrust_misc/thrust_misc.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/thrust_misc/thrust_misc.dp.cpp -o %T/thrust_misc/thrust_misc.dp.o %}

#include <iostream>
#include <thrust/advance.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>


// Unary function object for transforming input values
struct Square {
  __host__ __device__ int operator()(int x) const { return x * x; }
};


void test_1() {
  const int N = 5;
  thrust::device_vector<int> vec(N);
  thrust::sequence(vec.begin(), vec.end());
  thrust::device_vector<int>::iterator iter = vec.begin();

  // CHECK:  auto output_iter = dpct::make_transform_output_iterator(vec.begin(), Square());
  // CHECK-NEXT:  std::advance(iter, 2);
  // CHECK-NEXT:  std::transform(oneapi::dpl::execution::make_device_policy(q_ct1), vec.begin(), vec.end(), output_iter, Square());
  auto output_iter = thrust::make_transform_output_iterator(vec.begin(), Square());
  thrust::advance(iter, 2);
  thrust::transform(vec.begin(), vec.end(), output_iter, Square());
}
