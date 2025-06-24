#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <thrust/adjacent_difference.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>

void inclusive_scan_test() {
  // clang-format off
  // Start
  thrust::device_vector<int> A(4);
  std::vector<int> B(4);
  thrust::device_vector<int> R(4);
  std::vector<int> R2(4);
  /*1*/ thrust::inclusive_scan(B.begin(), B.end(), R2.begin(), thrust::minus<int>());
  /*2*/ thrust::inclusive_scan(thrust::device, A.begin(), A.end(), R.begin(),thrust::minus<int>());
  /*3*/ thrust::inclusive_scan(A.begin(), A.end(), R.begin());
  /*4*/ thrust::inclusive_scan(thrust::device, A.begin(), A.end(), R.begin());
  // End
  // clang-format on
}
