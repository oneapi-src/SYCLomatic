#include <thrust/adjacent_difference.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>

void thrust_not2() {
  // clang-format off
  // Start
  struct greater_than_zero {
    __host__ __device__ bool operator()(int x) const { return x > 0; }
    typedef int argument_type;
  };
  greater_than_zero pred;
  thrust::not2(thrust::greater_equal<int>());
  // End
  // clang-format on
}
