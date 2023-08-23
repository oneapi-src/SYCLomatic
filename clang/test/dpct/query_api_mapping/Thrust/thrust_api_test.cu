// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::set_union_by_key | FileCheck %s -check-prefix=set_union_by_key

// set_union_by_key:  dpct::set_union(oneapi::dpl::execution::seq, A_keys, A_keys + 3, B_keys, B_keys + 4, A_vals, B_vals, keys_result, vals_result);
// set_union_by_key-NEXT:  dpct::set_union(oneapi::dpl::execution::seq, A_keys, A_keys + 3, B_keys, B_keys + 4, A_vals, B_vals, keys_result, vals_result);
// set_union_by_key-NEXT:  dpct::set_union(oneapi::dpl::execution::seq, A_keys, A_keys + 7, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result, Compare());
// set_union_by_key-NEXT:  dpct::set_union(oneapi::dpl::execution::seq, A_keys, A_keys + 7, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result, Compare());
// set_union_by_key-NEXT:  dpct::set_union(oneapi::dpl::execution::make_device_policy(q_ct1), d_A_keys.begin(), d_A_keys.end(), d_B_keys.begin(), d_B_keys.end(), d_A_vals.begin(), d_B_vals.begin(), d_keys_result.begin(), d_vals_result.begin());
// set_union_by_key-NEXT:  dpct::set_union(oneapi::dpl::execution::make_device_policy(q_ct1), d_A_keys.begin(), d_A_keys.end(), d_B_keys.begin(), d_B_keys.end(), d_A_vals.begin(), d_B_vals.begin(), d_keys_result.begin(), d_vals_result.begin());
// set_union_by_key-NEXT:  dpct::set_union(oneapi::dpl::execution::make_device_policy(q_ct1), d_A_keys.begin(), d_A_keys.end(), d_B_keys.begin(), d_B_keys.end(), d_A_vals.begin(), d_B_vals.begin(), d_keys_result.begin(), d_vals_result.begin(), Compare());
// set_union_by_key-NEXT:  dpct::set_union(oneapi::dpl::execution::make_device_policy(q_ct1), d_A_keys.begin(), d_A_keys.end(), d_B_keys.begin(), d_B_keys.end(), d_A_vals.begin(), d_B_vals.begin(), d_keys_result.begin(), d_vals_result.begin(), Compare());
// set_union_by_key-NEXT:  dpct::set_union(oneapi::dpl::execution::seq, h_A_keys.begin(), h_A_keys.end(), h_B_keys.begin(), h_B_keys.end(), h_A_vals.begin(), h_B_vals.begin(), h_keys_result.begin(), h_vals_result.begin());
// set_union_by_key-NEXT:  dpct::set_union(oneapi::dpl::execution::seq, h_A_keys.begin(), h_A_keys.end(), h_B_keys.begin(), h_B_keys.end(), h_A_vals.begin(), h_B_vals.begin(), h_keys_result.begin(), h_vals_result.begin());
// set_union_by_key-NEXT:  dpct::set_union(oneapi::dpl::execution::seq, h_A_keys.begin(), h_A_keys.end(), h_B_keys.begin(), h_B_keys.end(), h_A_vals.begin(), h_B_vals.begin(), h_keys_result.begin(), h_vals_result.begin(), Compare());
// set_union_by_key-NEXT:  dpct::set_union(oneapi::dpl::execution::seq, h_A_keys.begin(), h_A_keys.end(), h_B_keys.begin(), h_B_keys.end(), h_A_vals.begin(), h_B_vals.begin(), h_keys_result.begin(), h_vals_result.begin(), Compare());
// set_union_by_key-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::set_union | FileCheck %s -check-prefix=set_union
// set_union:  oneapi::dpl::set_union(oneapi::dpl::execution::seq, A1, A1 + 7, A2, A2 + 5, result);
// set_union-NEXT:  oneapi::dpl::set_union(oneapi::dpl::execution::seq, A1, A1 + 7, A2, A2 + 5, result);
// set_union-NEXT:  oneapi::dpl::set_union(oneapi::dpl::execution::seq, A1, A1 + 7, A2, A2 + 5, result, std::greater<int>());
// set_union-NEXT:  oneapi::dpl::set_union(oneapi::dpl::execution::seq, A1, A1 + 7, A2, A2 + 5, result, std::greater<int>());
// set_union-NEXT:  oneapi::dpl::set_union(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin());
// set_union-NEXT:  oneapi::dpl::set_union(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin());
// set_union-NEXT:  oneapi::dpl::set_union(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin(), std::greater<int>());
// set_union-NEXT:  oneapi::dpl::set_union(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin(), std::greater<int>());
// set_union-NEXT:  oneapi::dpl::set_union(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin());
// set_union-NEXT:  oneapi::dpl::set_union(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin());
// set_union-NEXT:  oneapi::dpl::set_union(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin(), std::greater<int>());
// set_union-NEXT:  oneapi::dpl::set_union(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin(), std::greater<int>());
// set_union-EMPTY:
