#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>

void thrust_make_transform_iterator() {
  // clang-format off
  // Start
  thrust::device_vector<int> d_vec(10);
  auto iter = thrust::make_reverse_iterator(d_vec.begin());
  // End
  // clang-format on
}
