#include <thrust/adjacent_difference.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
// for cuda 12.0
#include <thrust/execution_policy.h>
#include <thrust/remove.h>
void replace() {
  // clang-format off
  // Start
  thrust::device_vector<int> A(4);
  thrust::device_vector<int> B(4);
  /*1*/ thrust::replace(A.begin(), A.end(), 0, 399);
  /*2*/ thrust::replace(B.begin(), B.end(), 0, 399);
  /*3*/ thrust::replace(thrust::device, A.begin(), A.end(), 0, 399);
  /*4*/ thrust::replace(thrust::device, B.begin(), B.end(), 0, 399);
  // End
  // clang-format on
}
