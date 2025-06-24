#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

void copy_if() {
  // clang-format off
  // Start
  auto is_even = [] __host__ __device__(int v) { return (v % 2) == 0; };
  const int N = 4;
  int vec_host_in[N] = {-1, 0, 1, 2};
  int vec_host_out[N];
  int *vec_devic_in;
  int *vec_devic_out;
  thrust::device_vector<int> dVecIn(vec_host_in, vec_host_in + N);
  thrust::device_vector<int> dVecOut(N);
  /*1*/ thrust::copy_if(vec_host_in, vec_host_in + N, vec_host_out, is_even);
  /*2*/ thrust::copy_if(thrust::device, vec_devic_in, vec_devic_in + N, vec_devic_out, is_even);
  /*3*/ thrust::copy_if(thrust::host, vec_host_in, vec_host_in + N,vec_host_out, is_even);
  /*4*/ thrust::copy_if(dVecIn.begin(), dVecIn.end(), dVecOut.begin(), is_even);
  // End
  // clang-format on
}
