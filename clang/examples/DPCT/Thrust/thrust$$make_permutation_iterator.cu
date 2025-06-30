#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/scan.h>

void thrust_make_permutation_iterator() {

  // clang-format off
  // Start
  thrust::host_vector<int> h_input(10);
  thrust::host_vector<int> h_input2(10);
  thrust::make_permutation_iterator(h_input.begin(),h_input2.begin());
  // End
  // clang-format on
}
