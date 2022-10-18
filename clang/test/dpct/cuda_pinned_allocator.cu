// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none --usm-level=none -out-root %T/cuda_pinned_alloc %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --match-full-lines --input-file %T/cuda_pinned_alloc/cuda_pinned_allocator.dp.cpp %s


#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

#define SIZE 4

int main(int argc, char *argv[])
{
    // CHECK: std::vector<float, dpct::deprecated::usm_host_allocator<float>> hVec(SIZE);
    std::vector<float, thrust::system::cuda::experimental::pinned_allocator<float>> hVec(SIZE);
    std::fill(hVec.begin(), hVec.end(), 2);

    // CHECK: dpct::device_vector<float> dVec(hVec.size());
    thrust::device_vector<float> dVec(hVec.size());
    // CHECK: std::copy(oneapi::dpl::execution::make_device_policy(q_ct1), hVec.begin(), hVec.end(), dVec.begin());
    thrust::copy(hVec.begin(), hVec.end(), dVec.begin());

    // CHECK: std::transform(oneapi::dpl::execution::make_device_policy(q_ct1), dVec.begin(), dVec.end(), dpct::make_constant_iterator(2), dVec.begin(), std::multiplies<float>());
    thrust::transform(dVec.begin(), dVec.end(), thrust::make_constant_iterator(2), dVec.begin(), thrust::multiplies<float>());
    // CHECK: std::copy(oneapi::dpl::execution::make_device_policy(q_ct1), dVec.begin(), dVec.end(), hVec.begin());
    thrust::copy(dVec.begin(), dVec.end(), hVec.begin());

    // CHECK: std::vector<float, dpct::deprecated::usm_host_allocator<float>> hVecCopy = hVec;
    std::vector<float, thrust::cuda::experimental::pinned_allocator<float>> hVecCopy = hVec;
    for (const auto &v : hVecCopy)
    {
        assert(v == 4 && "hVec elemenet should equal 4");
        std::cout<<v<<std::endl;
    }
    return 0;
}
