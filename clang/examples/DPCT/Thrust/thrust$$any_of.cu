#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <thrust/adjacent_difference.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>

void any_of_test() {
  // clang-format off
  // Start
  struct greater_than_zero {
    __host__ __device__ bool operator()(int x) const { return x > 0; }
  };
  greater_than_zero pred;
  thrust::device_vector<int> A(4);
  thrust::device_vector<int> B(4);
  /*1*/ thrust::any_of(A.begin(), A.end(), pred);
  /*2*/ thrust::any_of(thrust::device, B.begin(), B.end(), pred);
  // End
  // clang-format on
}
