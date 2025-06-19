#include <thrust/advance.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

void thrust_make_transform_output_iterator() {
  // clang-format off
  // Start
  struct square {
    __host__ __device__ int operator()(int x) const { return x * x; }
  };
  const int N = 5;
  thrust::device_vector<int> vec(N);
  auto output_iter = thrust::make_transform_output_iterator(vec.begin(), square());
    // End
  // clang-format on
}
