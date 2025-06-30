#include <cuda.h>
#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

void for_each_test() {
  // clang-format off
  // Start
  struct add_functor {
    __host__ __device__ void operator()(int &x) { x++; }
  };
  auto loop_body = [=] __device__ __host__(int ind) -> void {};
  thrust::device_vector<int> t;
  /*1*/ thrust::for_each(t.begin(), t.end(), add_functor());
  /*2*/ thrust::for_each(thrust::cuda::par_nosync, t.begin(), t.end(), loop_body );
  // End
  // clang-format on
}
