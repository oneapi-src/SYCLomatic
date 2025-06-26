#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

void transform() {
  // clang-format off
  // Start
  const int N = 1000;
  thrust::device_vector<float> t1(N);
  thrust::device_vector<float> t2(N);
  thrust::device_vector<float> t3(N);
  thrust::transform(t1.begin(), t1.end(), t2.begin(), t3.begin(), thrust::divides<float>());
  // End
  // clang-format on
}
