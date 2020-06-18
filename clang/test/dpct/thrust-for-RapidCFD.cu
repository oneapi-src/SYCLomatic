// RUN: dpct -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/thrust-for-RapidCFD.dp.cpp --match-full-lines %s

// CHECK:#include <CL/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <dpct/dpstd_utils.hpp>
// CHECK-NEXT: #include <dpstd/execution>
// CHECK-NEXT: #include <dpstd/algorithm>
// CHECK-NEXT: #include <numeric>
#include <thrust/scan.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>

int main(){
    return 0;
}
template<class T = int>
class MyClass{
    T t;
};

void foo_host(){

    thrust::host_vector<int> h_input(10);
    thrust::host_vector<int> h_input2(10);
    thrust::host_vector<int> h_output(10);
    thrust::host_vector<int> h_output2(10);

    //type
    //CHECK: dpstd::identity();
    thrust::identity<int>();
    //CHECK: MyClass<dpstd::identity> M;
    MyClass<thrust::identity<int> > M;
    //CHECK: MyClass<> M2;
    MyClass<thrust::use_default> M2;

    //iterator
    //CHECK: dpstd::make_permutation_iterator(h_input.begin(), h_input2.begin());
    thrust::make_permutation_iterator(h_input.begin(),h_input2.begin());
    //CHECK: dpstd::make_transform_iterator(h_input.begin(), std::negate<int>());
    thrust::make_transform_iterator(h_input.begin(), thrust::negate<int>());

    //functor
    //CHECK: std::minus<int>();
    thrust::minus<int>();
    //CHECK: std::negate<int>();
    thrust::negate<int>();
    //CHECK: std::logical_or<bool>();
    thrust::logical_or<bool>();

    //algo
    //CHECK: std::uninitialized_fill(h_input.begin(), h_input.end(), 10);
    thrust::uninitialized_fill(h_input.begin(), h_input.end(), 10);
    //CHECK: std::unique(h_input.begin(), h_input.end());
    thrust::unique(h_input.begin(), h_input.end());
    //CHECK: std::exclusive_scan(dpstd::execution::make_device_policy(q_ct1), h_input.begin(), h_input.end(), h_output.begin(), 0);
    thrust::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin());
    //CHECK: std::max_element(h_input.begin(), h_input.end());
    thrust::max_element(h_input.begin(), h_input.end());
    //CHECK: std::min_element(h_input.begin(), h_input.end());
    thrust::min_element(h_input.begin(), h_input.end());

    //CHECK: dpct::discard_iterator();
    thrust::make_discard_iterator();
    //CHECK: dpstd::reduce_by_segment(
    //CHECK-NEXT:     dpstd::execution::make_device_policy(q_ct1), h_input.begin(),
    //CHECK-NEXT:     h_input.end(),
    //CHECK-NEXT:     h_input2.begin(),
    //CHECK-NEXT:     h_output.begin(),
    //CHECK-NEXT:     h_output2.begin());
    thrust::reduce_by_key
    (
        h_input.begin(),
        h_input.end(),
        h_input2.begin(),
        h_output.begin(),
        h_output2.begin()
    );
}