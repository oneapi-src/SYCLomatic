// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: c2s --format-range=none -out-root %T/thrust-transform-if %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -fno-delayed-template-parsing -std=c++17 -fsized-deallocation
// RUN: FileCheck --input-file %T/thrust-transform-if/thrust-transform-if.dp.cpp --match-full-lines %s
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

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

int main() {
  const int dataLen = 10;
  int inDataH[dataLen]  = {-5, 0, 2, -3, 2, 4, 0, -1, 2, 8};
  int outDataH[dataLen];
  int stencilH[dataLen] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};

  thrust::device_vector<int> inDataD(dataLen);
  thrust::device_vector<int> outDataD(dataLen);
  thrust::device_vector<int> stencilD(dataLen);

  // Policy
// CHECK:  c2s::transform_if(oneapi::dpl::execution::seq, inDataH, inDataH + dataLen, outDataH, neg, is_odd());
  thrust::transform_if(thrust::host, inDataH, inDataH + dataLen, outDataH, neg, is_odd());

  // Policy and stencil
// CHECK:  c2s::transform_if(oneapi::dpl::execution::seq, inDataH, inDataH + dataLen, stencilH, outDataH, neg, identity());
  thrust::transform_if(thrust::host, inDataH, inDataH + dataLen, stencilH, outDataH, neg, identity());

  // Policy, second input, stencil and binary op
// CHECK:  c2s::transform_if(oneapi::dpl::execution::seq, inDataH, inDataH + dataLen, inDataH, stencilH, outDataH, plus, identity());
  thrust::transform_if(thrust::host, inDataH, inDataH + dataLen, inDataH, stencilH, outDataH, plus, identity());

  // No policy
// CHECK:  c2s::transform_if(oneapi::dpl::execution::make_device_policy(q_ct1), inDataD.begin(), inDataD.end(), outDataD.begin(), neg, is_odd());
  thrust::transform_if(inDataD.begin(), inDataD.end(), outDataD.begin(), neg, is_odd());

  // No policy and stencil
// CHECK:  c2s::transform_if(oneapi::dpl::execution::make_device_policy(q_ct1), inDataD.begin(), inDataD.end(), stencilD.begin(), outDataD.begin(), neg, identity());
  thrust::transform_if(inDataD.begin(), inDataD.end(), stencilD.begin(), outDataD.begin(), neg, identity());

  // No policy, second input, stencil and binary op
// CHECK:  c2s::transform_if(oneapi::dpl::execution::make_device_policy(q_ct1), inDataD.begin(), inDataD.end(), inDataD.begin(), stencilD.begin(), outDataD.begin(), plus, identity());
  thrust::transform_if(inDataD.begin(), inDataD.end(), inDataD.begin(), stencilD.begin(), outDataD.begin(), plus, identity());
}
