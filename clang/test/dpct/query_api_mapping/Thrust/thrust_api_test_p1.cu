// UNSUPPORTED: system-windows
// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.2, cuda-11.8

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::unique_count --extra-arg="-std=c++14"| FileCheck %s -check-prefix=unique_count --match-full-lines
// unique_count:  /*1*/ count = dpct::unique_count(oneapi::dpl::execution::seq, A, A + N, oneapi::dpl::equal_to<int>());
// unique_count-NEXT:  /*2*/ count = dpct::unique_count(oneapi::dpl::execution::seq, A, A + N, oneapi::dpl::equal_to<int>());
// unique_count-NEXT:  /*3*/ count = dpct::unique_count(oneapi::dpl::execution::seq, A, A + N);
// unique_count-NEXT:  /*4*/ count = dpct::unique_count(oneapi::dpl::execution::seq, A, A + N);
// unique_count-NEXT:  /*5*/ count = dpct::unique_count(oneapi::dpl::execution::seq, h_A.begin(), h_A.begin() + N, oneapi::dpl::equal_to<int>());
// unique_count-NEXT:  /*6*/ count = dpct::unique_count(oneapi::dpl::execution::make_device_policy(q_ct1), d_A.begin(), d_A.begin() + N, oneapi::dpl::equal_to<int>());
// unique_count-NEXT:  /*7*/ count = dpct::unique_count(oneapi::dpl::execution::seq, h_A.begin(), h_A.begin() + N, oneapi::dpl::equal_to<int>());
// unique_count-NEXT:  /*8*/ count = dpct::unique_count(oneapi::dpl::execution::make_device_policy(q_ct1), d_A.begin(), d_A.begin() + N, oneapi::dpl::equal_to<int>());
// unique_count-NEXT:  /*9*/ count =
// unique_count-NEXT:      dpct::unique_count(oneapi::dpl::execution::seq, h_A.begin(), h_A.begin() + N);
// unique_count-NEXT:  /*10*/ count =
// unique_count-NEXT:      dpct::unique_count(oneapi::dpl::execution::make_device_policy(q_ct1), d_A.begin(), d_A.begin() + N);
// unique_count-NEXT:  /*11*/ count = dpct::unique_count(oneapi::dpl::execution::seq, h_A.begin(), h_A.begin() + N);
// unique_count-NEXT:  /*12*/ count = dpct::unique_count(oneapi::dpl::execution::make_device_policy(q_ct1), d_A.begin(), d_A.begin() + N);
