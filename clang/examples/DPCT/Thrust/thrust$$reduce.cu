#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

void reduce_test() {
  // clang-format off
  // Start
  int data[6] = {1, 0, 2, 2, 1, 3};
  thrust::device_vector<int> d_data(data, data + 6);
  int result;
  /*1*/ result = thrust::reduce(thrust::device, d_data.begin(), d_data.begin() + 6);
  /*2*/ result = thrust::reduce(thrust::device, d_data.begin(), d_data.begin() + 6, 1);
  /*3*/ result = thrust::reduce(d_data.begin(), d_data.begin() + 6, 1);
  /*4*/ result = thrust::reduce(thrust::device, d_data.begin(), d_data.begin() + 6, -1, thrust::maximum<int>());
  /*5*/ result = thrust::reduce(d_data.begin(), d_data.begin() + 6, -1, thrust::maximum<int>());
  // End
  // clang-format on
}
