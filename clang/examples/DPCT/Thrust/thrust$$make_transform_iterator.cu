#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/scan.h>

void thrust_make_transform_iterator() {
  // clang-format off
  // Start
  thrust::host_vector<int> h_input(10);
  thrust::host_vector<int> h_input2(10);
  thrust::make_transform_iterator(h_input.begin(), thrust::negate<int>());
  // End
  // clang-format on
}
