#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <thrust/adjacent_difference.h>
// for cuda 12.0
#include <thrust/execution_policy.h>
#include <thrust/remove.h>

void remove_if_test() {
  // clang-format off
  // Start
  struct greater_than_zero {
    __host__ __device__ bool operator()(int x) const { return x > 0; }
  };
  thrust::device_vector<int> A(4);
  thrust::device_vector<int> S(4);
  greater_than_zero pred;
  thrust::device_vector<int> d;
  /*1*/ thrust::remove_if(thrust::device, A.begin(), A.end(), pred);
  /*2*/ thrust::remove_if(A.begin(), A.end(), pred);
  /*3*/ thrust::remove_if(thrust::device, A.begin(), A.end(), S.begin(), pred);
  /*4*/ thrust::remove_if(A.begin(), A.end(), S.begin(), pred);
  // End
  // clang-format on
}
