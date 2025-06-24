#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/replace.h>

void replace_copy_if() {
  // clang-format off
  // Start
  struct is_less_than_zero {
    __host__ __device__ bool operator()(int x) const { return x < 0; }
  };
  thrust::host_vector<int> AH(4);
  thrust::host_vector<int> BH(4);
  thrust::host_vector<int> SH(4);
  is_less_than_zero pred;
  /*1*/ thrust::replace_copy_if(AH.begin(), AH.end(), BH.begin(), pred, 0);
  /*2*/ thrust::replace_copy_if(AH.begin(), AH.end(), SH.begin(), BH.begin(), pred,0);
  /*3*/ thrust::replace_copy_if(thrust::host, AH.begin(), AH.end(), BH.begin(), pred,0);
  /*4*/ thrust::replace_copy_if(thrust::host, AH.begin(), AH.end(), SH.begin(),BH.begin(), pred, 0);
  // End
  // clang-format on
}
