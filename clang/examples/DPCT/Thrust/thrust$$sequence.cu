#include <thrust/advance.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

void sequence() {
  // clang-format off
  // Start
  const int N = 5;
  thrust::device_vector<int> vec(N);
  thrust::sequence(vec.begin(), vec.end());
  // End
  // clang-format on
}
