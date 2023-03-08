// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none --use-experimental-features=DPL_async -out-root %T/thrust-foeach-nosync %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only -fno-delayed-template-parsing -ferror-limit=50
// RUN: FileCheck --input-file %T/thrust-foeach-nosync/thrust-foeach-nosync.dp.cpp --match-full-lines %s
// CHECK: #include <oneapi/dpl/execution>
// CHECK-NEXT: #include <oneapi/dpl/algorithm>
// CHECK-NEXT: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <chrono>
// CHECK-NEXT: #include <dpct/dpl_utils.hpp>
#include <cuda.h>
#include <chrono>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>


int main() {
    
    
    const int n = 1e8;
    
    float* d_A;
    //CHECK: d_A = sycl::malloc_device<float>(n, dpct::get_default_queue());
    cudaMalloc( (void**) &d_A, n*sizeof(float));
    //CHECK: auto loop_begin = oneapi::dpl::counting_iterator<int>(0);
    auto loop_begin = thrust::counting_iterator<int>(0);
    auto loop_end = loop_begin+n;
    //CHECK: auto loop_body = [=] (int ind)-> void { d_A[ind] = sycl::cos((float)(1.0)); }; 
    auto loop_body = [=] __device__ __host__ (int ind)-> void { d_A[ind] = cosf(1.0); }; 

    for(int i=0;i<100;i++)
    {
      //CHECK: oneapi::dpl::experimental::for_each_async(oneapi::dpl::execution::par_unseq, loop_begin, loop_end, loop_body);
	    thrust::for_each(thrust::cuda::par_nosync, loop_begin, loop_end, loop_body );
    }
    //CHECK: dpct::get_current_device().queues_wait_and_throw();
    cudaDeviceSynchronize();    
    return 0;
}
