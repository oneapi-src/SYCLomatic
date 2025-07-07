#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/partition.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

void sort_test() {
  // clang-format off
  // Start
  thrust::host_vector<int> h_vec(10);
  thrust::device_vector<int> d_vec(10);
  /*1*/ thrust::sort(h_vec.begin(), h_vec.end());
  /*2*/ thrust::sort(thrust::device, d_vec.begin(), d_vec.end());
  // End
  // clang-format on
}
