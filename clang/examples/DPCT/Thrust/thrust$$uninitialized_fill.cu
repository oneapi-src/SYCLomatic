#include <thrust/scan.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/extrema.h>
#include <thrust/unique.h>

void uninitialized_fill() {
  // clang-format off
  // Start
  thrust::host_vector<int> h_input(10);
  thrust::uninitialized_fill(h_input.begin(), h_input.end(), 10);
  // End
  // clang-format on
}
