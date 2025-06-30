#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/tuple.h>
#include <thrust/device_ptr.h>
#include <thrust/count.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

void gather_if_test() {
  // clang-format off
  // Start
  thrust::host_vector<int> AH(4);
  thrust::host_vector<int> BH(4);
  thrust::host_vector<int> SH(4);
  thrust::host_vector<int> RH(4);
  struct is_less_than_zero {
    __host__ __device__ bool operator()(int x) const { return x < 0; }
  };
  is_less_than_zero pred;
  /*1*/ thrust::gather_if(AH.begin(), AH.end(), SH.begin(), BH.begin(), RH.begin(),pred);
  /*2*/ thrust::gather_if(thrust::host, AH.begin(), AH.end(), SH.begin(), BH.begin(),RH.begin(), pred);
  /*3*/ thrust::gather_if(AH.begin(), AH.end(), SH.begin(), BH.begin(), RH.begin());
  /*4*/ thrust::gather_if(thrust::host, AH.begin(), AH.end(), SH.begin(), BH.begin(),RH.begin());
  // End
  // clang-format on
}
