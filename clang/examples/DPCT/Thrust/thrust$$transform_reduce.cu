#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

auto square = [] __device__(int x) { return x * x; };

void transform_reduce() {
  // clang-format off
  // Start
  int data[10];
  cudaStream_t stream;
  thrust::device_ptr<int> begin = thrust::device_pointer_cast(&data[0]);
  thrust::device_ptr<int> end = begin + 10;
  /*1*/ bool h_result = thrust::transform_reduce(begin, end, square, 0, thrust::plus<bool>());
  /*2*/ bool h_result_1 = thrust::transform_reduce(thrust::seq, begin, end, square, 0, thrust::plus<bool>());
  /*3*/ bool h_result_2 = thrust::transform_reduce(thrust::cuda::par.on(stream), begin, end, square, 0, thrust::plus<bool>());
  // End
  // clang-format on
}
