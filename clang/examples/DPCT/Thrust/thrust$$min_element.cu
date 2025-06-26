#include <thrust/scan.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/extrema.h>
#include <thrust/unique.h>

void min_element_test() {
  // clang-format off
  // Start
  thrust::host_vector<int> h_input(10);
  thrust::min_element(h_input.begin(), h_input.end());
  // End
  // clang-format on
}
