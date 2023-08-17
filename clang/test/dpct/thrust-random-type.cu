// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T/thrust-random-type %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/thrust-random-type/thrust-random-type.dp.cpp --match-full-lines %s


#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/transform.h>

// CHECK:struct random_1 {
// CHECK-NEXT:  float operator()(const unsigned int n) const {
// CHECK-NEXT:    oneapi::dpl::default_engine rng;
// CHECK-NEXT:    oneapi::dpl::uniform_real_distribution<float> dist(1.0f, 2.0f);
// CHECK-NEXT:    rng.discard(n);
// CHECK-NEXT:    return dist(rng);
// CHECK-NEXT:  }
// CHECK-NEXT:};
struct random_1 {
  __host__ __device__ float operator()(const unsigned int n) const {
    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<float> dist(1.0f, 2.0f);
    rng.discard(n);
    return dist(rng);
  }
};


// CHECK:struct random_2 {
// CHECK-NEXT:  float operator()(const unsigned int n) const {
// CHECK-NEXT:    oneapi::dpl::default_engine rng;
// CHECK-NEXT:    rng.discard(n);
// CHECK-NEXT:    return (float)rng() / oneapi::dpl::default_engine::max();
// CHECK-NEXT:  }
// CHECK-NEXT:};
struct random_2 {
  __device__ float operator()(const unsigned int n) {
    thrust::default_random_engine rng;
    rng.discard(n);
    return (float)rng() / thrust::default_random_engine::max;
  }
};

void test(void) {
  {
    const int N = 20;
    // CHECK:    dpct::device_vector<float> numbers(N);
    // CHECK-NEXT:    oneapi::dpl::counting_iterator<unsigned int> index_sequence_begin(0);
    // CHECK-NEXT:    std::transform(oneapi::dpl::execution::par_unseq, index_sequence_begin, index_sequence_begin + N, numbers.begin(), random_1());
    // CHECK-NEXT:    std::transform(oneapi::dpl::execution::par_unseq, index_sequence_begin, index_sequence_begin + N, numbers.begin(), random_2());
    thrust::device_vector<float> numbers(N);
    thrust::counting_iterator<unsigned int> index_sequence_begin(0);
    thrust::transform(index_sequence_begin, index_sequence_begin + N, numbers.begin(), random_1());
    thrust::transform(index_sequence_begin, index_sequence_begin + N, numbers.begin(), random_2());
  }

  {
    // CHECK:    oneapi::dpl::uniform_int_distribution<int> dist1(-5, 10);
    // CHECK-NEXT:    oneapi::dpl::normal_distribution<float> dist2(1.0f, 2.0f);
    thrust::uniform_int_distribution<int> dist1(-5, 10);
    thrust::normal_distribution<float> dist2(1.0f, 2.0f);
  }
}
