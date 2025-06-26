#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

void copy_n() {
  // clang-format off
  // Start
  const char data[] = "abc";
  const size_t N = (sizeof(data) / sizeof(char)) - 1;
  char dst_data[N];
  thrust::device_vector<char> input(data, data + N);
  thrust::device_vector<char> dst(data, data + N);
  /*1*/ thrust::copy_n(data, N, dst_data);
  /*2*/ thrust::copy_n(thrust::device, input.begin(), N, dst.begin());
  // End
  // clang-format on
}
