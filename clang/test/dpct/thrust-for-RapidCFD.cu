// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct -out-root %T/thrust-for-RapidCFD %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17
// RUN: FileCheck --input-file %T/thrust-for-RapidCFD/thrust-for-RapidCFD.dp.cpp --match-full-lines %s

// CHECK: #include <oneapi/dpl/execution>
// CHECK-NEXT: #include <oneapi/dpl/algorithm>
// CHECK-NEXT:#include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <dpct/dpl_utils.hpp>
#include <thrust/scan.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
// for cuda 12.0
#include <thrust/extrema.h>
#include <thrust/unique.h>

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
    //CHECK: oneapi::dpl::identity();
    thrust::identity<int>();
    //CHECK: MyClass<oneapi::dpl::identity> M;
    MyClass<thrust::identity<int> > M;
    //CHECK: MyClass<> M2;
    MyClass<thrust::use_default> M2;

    //iterator
    //CHECK: oneapi::dpl::make_permutation_iterator(h_input.begin(), h_input2.begin());
    thrust::make_permutation_iterator(h_input.begin(),h_input2.begin());
    //CHECK: oneapi::dpl::make_transform_iterator(h_input.begin(), std::negate<int>());
    thrust::make_transform_iterator(h_input.begin(), thrust::negate<int>());

    //functor
    //CHECK: std::minus<int>();
    thrust::minus<int>();
    //CHECK: std::negate<int>();
    thrust::negate<int>();
    //CHECK: std::logical_or<bool>();
    thrust::logical_or<bool>();

    //algo
    //CHECK: std::uninitialized_fill(oneapi::dpl::execution::seq, h_input.begin(), h_input.end(), 10);
    thrust::uninitialized_fill(h_input.begin(), h_input.end(), 10);
    //CHECK: std::unique(oneapi::dpl::execution::seq, h_input.begin(), h_input.end());
    thrust::unique(h_input.begin(), h_input.end());
    //CHECK: std::exclusive_scan(oneapi::dpl::execution::seq, h_input.begin(), h_input.end(), h_output.begin(), (decltype(h_output.begin())::value_type)0);
    thrust::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin());
    //CHECK: std::max_element(oneapi::dpl::execution::seq, h_input.begin(), h_input.end());
    thrust::max_element(h_input.begin(), h_input.end());
    //CHECK: std::min_element(oneapi::dpl::execution::seq, h_input.begin(), h_input.end());
    thrust::min_element(h_input.begin(), h_input.end());

    //CHECK: oneapi::dpl::discard_iterator();
    thrust::make_discard_iterator();
    //CHECK: oneapi::dpl::reduce_by_segment(oneapi::dpl::execution::seq, h_input.begin(), h_input.end(), h_input2.begin(), h_output.begin(), h_output2.begin());
    thrust::reduce_by_key(h_input.begin(), h_input.end(), h_input2.begin(), h_output.begin(), h_output2.begin()
    );
}

