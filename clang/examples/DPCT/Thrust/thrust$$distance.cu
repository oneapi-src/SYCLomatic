#include <thrust/device_vector.h>
#include <thrust/distance.h>
#include <thrust/functional.h>

void thrust_proj() {
  // clang-format off
  // Start
  thrust::device_vector<int> vec(13);
  thrust::device_vector<int>::iterator iter1 = vec.begin();
  thrust::device_vector<int>::iterator iter2 = iter1 + 7;
  int d = thrust::distance(iter1, iter2);
  // End
  // clang-format on
}
