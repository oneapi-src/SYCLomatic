#include <thrust/scan.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/extrema.h>
#include <thrust/unique.h>

void unique() {
  // clang-format off
  // Start
  thrust::host_vector<int> h_input(10);
  thrust::unique(h_input.begin(), h_input.end());
  // End
  // clang-format on
}
