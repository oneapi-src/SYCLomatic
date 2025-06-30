#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

void stable_sort_by_key() {
  // clang-format off
  // Start
  struct is_odd {
    __host__ __device__ bool operator()(const int x) const {
      return x % 2;
    }
  };
  struct identity {
    __host__ __device__ bool operator()(const int x) const {
      return x;
    }
  };
  thrust::negate<int> neg;
  thrust::plus<int> plus;
  const int dataLen = 10;
  int inDataH[dataLen]  = {-5, 0, 2, -3, 2, 4, 0, -1, 2, 8};
  int outDataH[dataLen];
  int stencilH[dataLen] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  thrust::device_vector<int> inDataD(dataLen);
  thrust::device_vector<int> outDataD(dataLen);
  thrust::device_vector<int> stencilD(dataLen);
  /*1*/ thrust::transform_if(thrust::host, inDataH, inDataH + dataLen, outDataH, neg, is_odd());
  /*2*/ thrust::transform_if(thrust::host, inDataH, inDataH + dataLen, stencilH, outDataH, neg, identity());
  /*3*/ thrust::transform_if(thrust::host, inDataH, inDataH + dataLen, inDataH, stencilH, outDataH, plus, identity());
  /*4*/ thrust::transform_if(inDataD.begin(), inDataD.end(), outDataD.begin(), neg, is_odd());
  /*5*/ thrust::transform_if(inDataD.begin(), inDataD.end(), stencilD.begin(), outDataD.begin(), neg, identity());
  /*6*/ thrust::transform_if(inDataD.begin(), inDataD.end(), inDataD.begin(), stencilD.begin(), outDataD.begin(), plus, identity());
  // End
  // clang-format on
}
