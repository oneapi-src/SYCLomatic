#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

void stable_sort_by_key() {
  // clang-format off
  // Start
  thrust::device_vector<int> AD(4);
  thrust::device_vector<int> BD(4);
  thrust::host_vector<int> AH(4);
  thrust::host_vector<int> BH(4);
  /*1*/ thrust::stable_sort_by_key(                AH.begin(), AH.end(), BH.begin());
  /*2*/ thrust::stable_sort_by_key(                AH.begin(), AH.end(), BH.begin(), thrust::greater<int>());
  /*3*/ thrust::stable_sort_by_key(thrust::host,   AH.begin(), AH.end(), BH.begin());
  /*4*/ thrust::stable_sort_by_key(thrust::host,   AH.begin(), AH.end(), BH.begin(), thrust::greater<int>());
  // End
  // clang-format on
}
