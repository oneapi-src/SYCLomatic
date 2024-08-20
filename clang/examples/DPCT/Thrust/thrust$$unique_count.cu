#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
#include <thrust/partition.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

void unique_count() {
  const int N = 7;
  int count;
  int A[N] = {1, 3, 3, 3, 2, 2, 1};
  thrust::host_vector<int> h_A(A, A + N);
  thrust::device_vector<int> d_A(A, A + N);

  // clang-format off
  // Start
  /*1*/ count = thrust::unique_count(thrust::host, A, A + N, thrust::equal_to<int>());
  /*2*/ count = thrust::unique_count(A, A + N, thrust::equal_to<int>());
  /*3*/ count = thrust::unique_count(thrust::host, A, A + N);
  /*4*/ count = thrust::unique_count(A, A + N);
  /*5*/ count = thrust::unique_count(thrust::host, h_A.begin(), h_A.begin() + N,
                                     thrust::equal_to<int>());
  /*6*/ count = thrust::unique_count(thrust::device, d_A.begin(),
                                     d_A.begin() + N, thrust::equal_to<int>());
  /*7*/ count = thrust::unique_count(h_A.begin(), h_A.begin() + N,
                                     thrust::equal_to<int>());
  /*8*/ count = thrust::unique_count(d_A.begin(), d_A.begin() + N,
                                     thrust::equal_to<int>());
  /*9*/ count =
      thrust::unique_count(thrust::host, h_A.begin(), h_A.begin() + N);
  /*10*/ count =
      thrust::unique_count(thrust::device, d_A.begin(), d_A.begin() + N);
  /*11*/ count = thrust::unique_count(h_A.begin(), h_A.begin() + N);
  /*12*/ count = thrust::unique_count(d_A.begin(), d_A.begin() + N);
  // End
  // clang-format on
}
