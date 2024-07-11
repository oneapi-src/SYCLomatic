// UNSUPPORTED: cuda-8.0, cuda-9.0
// UNSUPPORTED: v8.0, v9.0
// RUN: dpct --format-range=none --use-experimental-features=dpl-experimental-api -out-root %T/thrust-for-each-nosync %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only -ferror-limit=50
// RUN: FileCheck --input-file %T/thrust-for-each-nosync/thrust-for-each-nosync.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/thrust-for-each-nosync/thrust-for-each-nosync.dp.cpp -o %T/thrust-for-each-nosync/thrust-for-each-nosync.dp.o %}
// CHECK: #include <oneapi/dpl/execution>
// CHECK-NEXT: #include <oneapi/dpl/algorithm>
// CHECK-NEXT: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <chrono>
// CHECK-NEXT: #include <dpct/dpl_utils.hpp>
// CHECK: #include <oneapi/dpl/async>
#include <cuda.h>
#include <chrono>
#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>


int main() {
    // CHECK: dpct::device_vector<float> dVec(4);
    thrust::device_vector<float> dVec(4);
    //CHECK: auto loop_body = [=] (int ind)-> void { }; 
    auto loop_body = [=] __device__ __host__ (int ind)-> void { }; 

    //CHECK: oneapi::dpl::experimental::for_each_async(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), dVec.begin(), dVec.end(), loop_body);
	  thrust::for_each(thrust::cuda::par_nosync, dVec.begin(), dVec.end(), loop_body );
    return 0;
}
