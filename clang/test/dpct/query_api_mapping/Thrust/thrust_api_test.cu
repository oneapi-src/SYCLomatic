// UNSUPPORTED: system-windows
// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::set_union_by_key --extra-arg="-std=c++14" | FileCheck %s -check-prefix=set_union_by_key
// set_union_by_key:  /*1*/ dpct::set_union(oneapi::dpl::execution::seq, A_keys, A_keys + 3, B_keys, B_keys + 4, A_vals, B_vals, keys_result, vals_result);
// set_union_by_key-NEXT:  /*2*/ dpct::set_union(oneapi::dpl::execution::seq, A_keys, A_keys + 3, B_keys, B_keys + 4, A_vals, B_vals, keys_result, vals_result);
// set_union_by_key-NEXT:  /*3*/ dpct::set_union(oneapi::dpl::execution::seq, A_keys, A_keys + 7, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result, Compare());
// set_union_by_key-NEXT:  /*4*/ dpct::set_union(oneapi::dpl::execution::seq, A_keys, A_keys + 7, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result, Compare());
// set_union_by_key-NEXT:  /*5*/ dpct::set_union(oneapi::dpl::execution::make_device_policy(q_ct1), d_A_keys.begin(), d_A_keys.end(), d_B_keys.begin(), d_B_keys.end(), d_A_vals.begin(), d_B_vals.begin(), d_keys_result.begin(), d_vals_result.begin());
// set_union_by_key-NEXT:  /*6*/ dpct::set_union(oneapi::dpl::execution::make_device_policy(q_ct1), d_A_keys.begin(), d_A_keys.end(), d_B_keys.begin(), d_B_keys.end(), d_A_vals.begin(), d_B_vals.begin(), d_keys_result.begin(), d_vals_result.begin());
// set_union_by_key-NEXT:  /*7*/ dpct::set_union(oneapi::dpl::execution::make_device_policy(q_ct1), d_A_keys.begin(), d_A_keys.end(), d_B_keys.begin(), d_B_keys.end(), d_A_vals.begin(), d_B_vals.begin(), d_keys_result.begin(), d_vals_result.begin(), Compare());
// set_union_by_key-NEXT:  /*8*/ dpct::set_union(oneapi::dpl::execution::make_device_policy(q_ct1), d_A_keys.begin(), d_A_keys.end(), d_B_keys.begin(), d_B_keys.end(), d_A_vals.begin(), d_B_vals.begin(), d_keys_result.begin(), d_vals_result.begin(), Compare());
// set_union_by_key-NEXT:  /*9*/ dpct::set_union(oneapi::dpl::execution::seq, h_A_keys.begin(), h_A_keys.end(), h_B_keys.begin(), h_B_keys.end(), h_A_vals.begin(), h_B_vals.begin(), h_keys_result.begin(), h_vals_result.begin());
// set_union_by_key-NEXT:  /*10*/ dpct::set_union(oneapi::dpl::execution::seq, h_A_keys.begin(), h_A_keys.end(), h_B_keys.begin(), h_B_keys.end(), h_A_vals.begin(), h_B_vals.begin(), h_keys_result.begin(), h_vals_result.begin());
// set_union_by_key-NEXT:  /*11*/ dpct::set_union(oneapi::dpl::execution::seq, h_A_keys.begin(), h_A_keys.end(), h_B_keys.begin(), h_B_keys.end(), h_A_vals.begin(), h_B_vals.begin(), h_keys_result.begin(), h_vals_result.begin(), Compare());
// set_union_by_key-NEXT:  /*12*/ dpct::set_union(oneapi::dpl::execution::seq, h_A_keys.begin(), h_A_keys.end(), h_B_keys.begin(), h_B_keys.end(), h_A_vals.begin(), h_B_vals.begin(), h_keys_result.begin(), h_vals_result.begin(), Compare());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::set_union --extra-arg="-std=c++14" | FileCheck %s -check-prefix=set_union
// set_union:  /*1*/ oneapi::dpl::set_union(oneapi::dpl::execution::seq, A1, A1 + 7, A2, A2 + 5, result);
// set_union-NEXT:  /*2*/ oneapi::dpl::set_union(oneapi::dpl::execution::seq, A1, A1 + 7, A2, A2 + 5, result);
// set_union-NEXT:  /*3*/ oneapi::dpl::set_union(oneapi::dpl::execution::seq, A1, A1 + 7, A2, A2 + 5, result, std::greater<int>());
// set_union-NEXT:  /*4*/ oneapi::dpl::set_union(oneapi::dpl::execution::seq, A1, A1 + 7, A2, A2 + 5, result, std::greater<int>());
// set_union-NEXT:  /*5*/ oneapi::dpl::set_union(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin());
// set_union-NEXT:  /*6*/ oneapi::dpl::set_union(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin());
// set_union-NEXT:  /*7*/ oneapi::dpl::set_union(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin(), std::greater<int>());
// set_union-NEXT:  /*8*/ oneapi::dpl::set_union(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin(), std::greater<int>());
// set_union-NEXT:  /*9*/ oneapi::dpl::set_union(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin());
// set_union-NEXT:  /*10*/ oneapi::dpl::set_union(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin());
// set_union-NEXT:  /*11*/ oneapi::dpl::set_union(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin(), std::greater<int>());
// set_union-NEXT:  /*12*/ oneapi::dpl::set_union(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin(), std::greater<int>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::set_intersection --extra-arg="-std=c++14"| FileCheck %s -check-prefix=set_intersection
// set_intersection:  /*1*/ oneapi::dpl::set_intersection(oneapi::dpl::execution::seq, A1, A1 + 6, A2, A2 + 7, result);
// set_intersection-NEXT:  /*2*/ oneapi::dpl::set_intersection(oneapi::dpl::execution::seq, A1, A1 + 6, A2, A2 + 7, result);
// set_intersection-NEXT:  /*3*/ oneapi::dpl::set_intersection(oneapi::dpl::execution::seq, A1, A1 + 6, A2, A2 + 7, result, std::greater<int>());
// set_intersection-NEXT:  /*4*/ oneapi::dpl::set_intersection(oneapi::dpl::execution::seq, A1, A1 + 6, A2, A2 + 7, result, std::greater<int>());
// set_intersection-NEXT:  /*5*/ oneapi::dpl::set_intersection(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin());
// set_intersection-NEXT:  /*6*/ oneapi::dpl::set_intersection(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin());
// set_intersection-NEXT:  /*7*/ oneapi::dpl::set_intersection(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin(), std::greater<int>());
// set_intersection-NEXT:  /*8*/ oneapi::dpl::set_intersection(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin(), std::greater<int>());
// set_intersection-NEXT:  /*9*/ oneapi::dpl::set_intersection(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin());
// set_intersection-NEXT:  /*10*/ oneapi::dpl::set_intersection(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin());
// set_intersection-NEXT:  /*11*/ oneapi::dpl::set_intersection(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin(), std::greater<int>());
// set_intersection-NEXT:  /*12*/ oneapi::dpl::set_intersection(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin(), std::greater<int>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::is_sorted_until --extra-arg="-std=c++14"| FileCheck %s -check-prefix=is_sorted_until
// is_sorted_until:  /*1*/ oneapi::dpl::is_sorted_until(oneapi::dpl::execution::seq, A, A + 8);
// is_sorted_until-NEXT:  /*2*/ oneapi::dpl::is_sorted_until(oneapi::dpl::execution::seq, A, A + 8);
// is_sorted_until-NEXT:  /*3*/ oneapi::dpl::is_sorted_until(oneapi::dpl::execution::seq, A, A + 8, comp);
// is_sorted_until-NEXT:  /*4*/ oneapi::dpl::is_sorted_until(oneapi::dpl::execution::seq, A, A + 8, comp);
// is_sorted_until-NEXT:  /*5*/ oneapi::dpl::is_sorted_until(oneapi::dpl::execution::seq, h_A.begin(), h_A.end());
// is_sorted_until-NEXT:  /*6*/ oneapi::dpl::is_sorted_until(oneapi::dpl::execution::seq, h_A.begin(), h_A.end());
// is_sorted_until-NEXT:  /*7*/ oneapi::dpl::is_sorted_until(oneapi::dpl::execution::seq, h_A.begin(), h_A.end(), comp);
// is_sorted_until-NEXT:  /*8*/ oneapi::dpl::is_sorted_until(oneapi::dpl::execution::seq, h_A.begin(), h_A.end(), comp);
// is_sorted_until-NEXT:  /*9*/ oneapi::dpl::is_sorted_until(oneapi::dpl::execution::make_device_policy(q_ct1), d_A.begin(), d_A.end());
// is_sorted_until-NEXT:  /*10*/ oneapi::dpl::is_sorted_until(oneapi::dpl::execution::make_device_policy(q_ct1), d_A.begin(), d_A.end());
// is_sorted_until-NEXT:  /*11*/ oneapi::dpl::is_sorted_until(oneapi::dpl::execution::make_device_policy(q_ct1), d_A.begin(), d_A.end(), comp);
// is_sorted_until-NEXT:  /*12*/ oneapi::dpl::is_sorted_until(oneapi::dpl::execution::make_device_policy(q_ct1), d_A.begin(), d_A.end(), comp);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::is_partitioned --extra-arg="-std=c++14"| FileCheck %s -check-prefix=is_partitioned
// is_partitioned:  /*1*/ oneapi::dpl::is_partitioned(oneapi::dpl::execution::seq, A, A + 10, is_even());
// is_partitioned-NEXT:  /*2*/ oneapi::dpl::is_partitioned(oneapi::dpl::execution::seq, A, A + 10, is_even());
// is_partitioned-NEXT:  /*3*/ oneapi::dpl::is_partitioned(oneapi::dpl::execution::seq, h_A.begin(), h_A.end(), is_even());
// is_partitioned-NEXT:  /*4*/ oneapi::dpl::is_partitioned(oneapi::dpl::execution::seq, h_A.begin(), h_A.end(), is_even());
// is_partitioned-NEXT:  /*5*/ oneapi::dpl::is_partitioned(oneapi::dpl::execution::make_device_policy(q_ct1), d_A.begin(), d_A.end(), is_even());
// is_partitioned-NEXT:  /*6*/ oneapi::dpl::is_partitioned(oneapi::dpl::execution::make_device_policy(q_ct1), d_A.begin(), d_A.end(), is_even());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::none_of --extra-arg="-std=c++14"| FileCheck %s -check-prefix=none_of
// none_of:   /*1*/ oneapi::dpl::none_of(oneapi::dpl::execution::seq, A, A + 2, oneapi::dpl::identity());
// none_of-NEXT:   /*2*/ oneapi::dpl::none_of(oneapi::dpl::execution::seq, A, A + 2, oneapi::dpl::identity());
// none_of-NEXT:   /*3*/ oneapi::dpl::none_of(oneapi::dpl::execution::seq, h_A.begin(), h_A.begin() + 2, oneapi::dpl::identity());
// none_of-NEXT:   /*4*/ oneapi::dpl::none_of(oneapi::dpl::execution::seq, h_A.begin(), h_A.begin() + 2, oneapi::dpl::identity());
// none_of-NEXT:   /*5*/ oneapi::dpl::none_of(oneapi::dpl::execution::make_device_policy(q_ct1), d_A.begin(), d_A.begin() + 2, oneapi::dpl::identity());
// none_of-NEXT:   /*6*/ oneapi::dpl::none_of(oneapi::dpl::execution::make_device_policy(q_ct1), d_A.begin(), d_A.begin() + 2, oneapi::dpl::identity());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::all_of --extra-arg="-std=c++14"| FileCheck %s -check-prefix=all_of
// all_of: /*1*/  oneapi::dpl::all_of(oneapi::dpl::execution::seq, A, A + 2, oneapi::dpl::identity());
// all_of-NEXT: /*2*/  oneapi::dpl::all_of(oneapi::dpl::execution::seq, A, A + 2, oneapi::dpl::identity());
// all_of-NEXT: /*3*/  oneapi::dpl::all_of(oneapi::dpl::execution::seq, h_A.begin(), h_A.begin() + 2, oneapi::dpl::identity());
// all_of-NEXT: /*4*/  oneapi::dpl::all_of(oneapi::dpl::execution::seq, h_A.begin(), h_A.begin() + 2, oneapi::dpl::identity());
// all_of-NEXT: /*5*/  oneapi::dpl::all_of(oneapi::dpl::execution::make_device_policy(q_ct1), d_A.begin(), d_A.begin() + 2, oneapi::dpl::identity());
// all_of-NEXT: /*6*/  oneapi::dpl::all_of(oneapi::dpl::execution::make_device_policy(q_ct1), d_A.begin(), d_A.begin() + 2, oneapi::dpl::identity());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::uninitialized_fill_n --extra-arg="-std=c++14"| FileCheck %s -check-prefix=uninitialized_fill_n
// uninitialized_fill_n:  /*1*/ oneapi::dpl::uninitialized_fill_n(oneapi::dpl::execution::make_device_policy(q_ct1), d_array, N, int_val);
// uninitialized_fill_n-NEXT:  /*2*/ oneapi::dpl::uninitialized_fill_n(oneapi::dpl::execution::make_device_policy(q_ct1), d_array, N, int_val);
// uninitialized_fill_n-NEXT:  /*3*/ oneapi::dpl::uninitialized_fill_n(oneapi::dpl::execution::seq, data, N, val);
// uninitialized_fill_n-NEXT:  /*4*/ oneapi::dpl::uninitialized_fill_n(oneapi::dpl::execution::seq, data, N, val);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::swap_ranges --extra-arg="-std=c++14"| FileCheck %s -check-prefix=swap_ranges
// swap_ranges:  /*1*/ oneapi::dpl::swap_ranges(oneapi::dpl::execution::make_device_policy(q_ct1), d_v1.begin(), d_v1.end(), d_v2.begin());
// swap_ranges-NEXT:  /*2*/ oneapi::dpl::swap_ranges(oneapi::dpl::execution::make_device_policy(q_ct1), d_v1.begin(), d_v1.end(), d_v2.begin());
// swap_ranges-NEXT:  /*3*/ oneapi::dpl::swap_ranges(oneapi::dpl::execution::seq, h_v1.begin(), h_v1.end(), h_v2.begin());
// swap_ranges-NEXT:  /*4*/ oneapi::dpl::swap_ranges(oneapi::dpl::execution::seq, h_v1.begin(), h_v1.end(), h_v2.begin());
// swap_ranges-NEXT:  /*5*/ oneapi::dpl::swap_ranges(oneapi::dpl::execution::seq, v1, v1 + 2, v2);
// swap_ranges-NEXT:  /*6*/ oneapi::dpl::swap_ranges(oneapi::dpl::execution::seq, v1, v1 + 2, v2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::set_symmetric_difference_by_key --extra-arg="-std=c++14"| FileCheck %s -check-prefix=set_symmetric_difference_by_key
// set_symmetric_difference_by_key:  /*1*/ dpct::set_symmetric_difference(oneapi::dpl::execution::seq, A_keys, A_keys + 7, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result);
// set_symmetric_difference_by_key-NEXT:  /*2*/ dpct::set_symmetric_difference(oneapi::dpl::execution::seq, A_keys, A_keys + 7, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result);
// set_symmetric_difference_by_key-NEXT:  /*3*/ dpct::set_symmetric_difference(oneapi::dpl::execution::seq, A_keys, A_keys + 7, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result, Compare());
// set_symmetric_difference_by_key-NEXT:  /*4*/ dpct::set_symmetric_difference(oneapi::dpl::execution::seq, A_keys, A_keys + 7, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result, Compare());
// set_symmetric_difference_by_key-NEXT:  /*5*/ dpct::set_symmetric_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_A_keys.begin(), d_A_keys.end(), d_B_keys.begin(), d_B_keys.end(), d_A_vals.begin(), d_B_vals.begin(), d_keys_result.begin(), d_vals_result.begin());
// set_symmetric_difference_by_key-NEXT:  /*6*/ dpct::set_symmetric_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_A_keys.begin(), d_A_keys.end(), d_B_keys.begin(), d_B_keys.end(), d_A_vals.begin(), d_B_vals.begin(), d_keys_result.begin(), d_vals_result.begin());
// set_symmetric_difference_by_key-NEXT:  /*7*/ dpct::set_symmetric_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_A_keys.begin(), d_A_keys.end(), d_B_keys.begin(), d_B_keys.end(), d_A_vals.begin(), d_B_vals.begin(), d_keys_result.begin(), d_vals_result.begin(), Compare());
// set_symmetric_difference_by_key-NEXT:  /*8*/ dpct::set_symmetric_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_A_keys.begin(), d_A_keys.end(), d_B_keys.begin(), d_B_keys.end(), d_A_vals.begin(), d_B_vals.begin(), d_keys_result.begin(), d_vals_result.begin(), Compare());
// set_symmetric_difference_by_key-NEXT:  /*9*/ dpct::set_symmetric_difference(oneapi::dpl::execution::seq, h_A_keys.begin(), h_A_keys.end(), h_B_keys.begin(), h_B_keys.end(), h_A_vals.begin(), h_B_vals.begin(), h_keys_result.begin(), h_vals_result.begin());
// set_symmetric_difference_by_key-NEXT:  /*10*/ dpct::set_symmetric_difference(oneapi::dpl::execution::seq, h_A_keys.begin(), h_A_keys.end(), h_B_keys.begin(), h_B_keys.end(), h_A_vals.begin(), h_B_vals.begin(), h_keys_result.begin(), h_vals_result.begin());
// set_symmetric_difference_by_key-NEXT:  /*11*/ dpct::set_symmetric_difference(oneapi::dpl::execution::seq, h_A_keys.begin(), h_A_keys.end(), h_B_keys.begin(), h_B_keys.end(), h_A_vals.begin(), h_B_vals.begin(), h_keys_result.begin(), h_vals_result.begin(), Compare());
// set_symmetric_difference_by_key-NEXT:  /*12*/ dpct::set_symmetric_difference(oneapi::dpl::execution::seq, h_A_keys.begin(), h_A_keys.end(), h_B_keys.begin(), h_B_keys.end(), h_A_vals.begin(), h_B_vals.begin(), h_keys_result.begin(), h_vals_result.begin(), Compare());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::set_symmetric_difference --extra-arg="-std=c++14"| FileCheck %s -check-prefix=set_symmetric_difference
// set_symmetric_difference:  /*1*/ oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::seq, A1, A1 + 4, A2, A2 + 2, result);
// set_symmetric_difference-NEXT:  /*2*/ oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::seq, A1, A1 + 4, A2, A2 + 2, result);
// set_symmetric_difference-NEXT:  /*3*/ oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::seq, A1, A1 + 5, A2, A2 + 5, result, Compare());
// set_symmetric_difference-NEXT:  /*4*/ oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::seq, A1, A1 + 5, A2, A2 + 5, result, Compare());
// set_symmetric_difference-NEXT:  /*5*/ oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin());
// set_symmetric_difference-NEXT:  /*6*/ oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin());
// set_symmetric_difference-NEXT:  /*7*/ oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin(), oneapi::dpl::less<int>());
// set_symmetric_difference-NEXT:  /*8*/ oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin(), oneapi::dpl::less<int>());
// set_symmetric_difference-NEXT:  /*9*/ oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin());
// set_symmetric_difference-NEXT:  /*10*/ oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin());
// set_symmetric_difference-NEXT:  /*11*/ oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin(), oneapi::dpl::less<int>());
// set_symmetric_difference-NEXT:  /*12*/ oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin(), oneapi::dpl::less<int>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::equal --extra-arg="-std=c++14"| FileCheck %s -check-prefix=equal
// equal:  /*1*/ oneapi::dpl::equal(oneapi::dpl::execution::seq, A1, A1 + N, A2);
// equal-NEXT:  /*2*/ oneapi::dpl::equal(oneapi::dpl::execution::seq, A1, A1 + N, A2);
// equal-NEXT:  /*3*/ oneapi::dpl::equal(oneapi::dpl::execution::seq, x, x + N, y, compare_modulo_two());
// equal-NEXT:  /*4*/ oneapi::dpl::equal(oneapi::dpl::execution::seq, x, x + N, y, compare_modulo_two());
// equal-NEXT:  /*5*/ oneapi::dpl::equal(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin());
// equal-NEXT:  /*6*/ oneapi::dpl::equal(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin());
// equal-NEXT:  /*7*/ oneapi::dpl::equal(oneapi::dpl::execution::seq, h_x.begin(), h_x.end(), h_y.begin(), compare_modulo_two());
// equal-NEXT:  /*8*/ oneapi::dpl::equal(oneapi::dpl::execution::seq, h_x.begin(), h_x.end(), h_y.begin(), compare_modulo_two());
// equal-NEXT:  /*9*/ oneapi::dpl::equal(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin());
// equal-NEXT:  /*10*/ oneapi::dpl::equal(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin());
// equal-NEXT:  /*11*/ oneapi::dpl::equal(oneapi::dpl::execution::make_device_policy(q_ct1), d_x.begin(), d_x.end(), d_y.begin(), compare_modulo_two());
// equal-NEXT:  /*12*/ oneapi::dpl::equal(oneapi::dpl::execution::make_device_policy(q_ct1), d_x.begin(), d_x.end(), d_y.begin(), compare_modulo_two());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::uninitialized_copy_n --extra-arg="-std=c++14"| FileCheck %s -check-prefix=uninitialized_copy_n
// uninitialized_copy_n:  /*1*/ oneapi::dpl::uninitialized_copy_n(oneapi::dpl::execution::make_device_policy(q_ct1), d_input.begin(), N, d_array);
// uninitialized_copy_n-NEXT:  /*2*/ oneapi::dpl::uninitialized_copy_n(oneapi::dpl::execution::seq, h_data, N, h_array);
// uninitialized_copy_n-NEXT:  /*3*/ oneapi::dpl::uninitialized_copy_n(oneapi::dpl::execution::make_device_policy(q_ct1), d_input.begin(), N, d_array);
// uninitialized_copy_n-NEXT:  /*4*/ oneapi::dpl::uninitialized_copy_n(oneapi::dpl::execution::seq, h_data, N, h_array);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::uninitialized_copy --extra-arg="-std=c++14"| FileCheck %s -check-prefix=uninitialized_copy
// uninitialized_copy:  /*1*/ oneapi::dpl::uninitialized_copy(oneapi::dpl::execution::make_device_policy(q_ct1), d_input.begin(), d_input.end(), d_array);
// uninitialized_copy-NEXT:  /*2*/ oneapi::dpl::uninitialized_copy(oneapi::dpl::execution::seq, data, data + N, array);
// uninitialized_copy-NEXT:  /*3*/ oneapi::dpl::uninitialized_copy(oneapi::dpl::execution::make_device_policy(q_ct1), d_input.begin(), d_input.end(), d_array);
// uninitialized_copy-NEXT:  /*4*/ oneapi::dpl::uninitialized_copy(oneapi::dpl::execution::seq, data, data + N, h_array);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::transform_inclusive_scan --extra-arg="-std=c++14"| FileCheck %s -check-prefix=transform_inclusive_scan
// transform_inclusive_scan:  /*1*/ oneapi::dpl::transform_inclusive_scan(oneapi::dpl::execution::seq, data, data + N, data, binary_op, unary_op);
// transform_inclusive_scan-NEXT:  /*2*/ oneapi::dpl::transform_inclusive_scan(oneapi::dpl::execution::seq, h_vec_data.begin(), h_vec_data.end(), h_vec_data.begin(), binary_op, unary_op);
// transform_inclusive_scan-NEXT:  /*3*/ oneapi::dpl::transform_inclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), d_vec_data.begin(), d_vec_data.end(), d_vec_data.begin(), binary_op, unary_op);
// transform_inclusive_scan-NEXT:  /*4*/ oneapi::dpl::transform_inclusive_scan(oneapi::dpl::execution::seq, data, data + N, data, binary_op, unary_op);
// transform_inclusive_scan-NEXT:  /*5*/ oneapi::dpl::transform_inclusive_scan(oneapi::dpl::execution::seq, h_vec_data.begin(), h_vec_data.end(), h_vec_data.begin(), binary_op, unary_op);
// transform_inclusive_scan-NEXT:  /*6*/ oneapi::dpl::transform_inclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), d_vec_data.begin(), d_vec_data.end(), d_vec_data.begin(), binary_op, unary_op);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::equal_range --extra-arg="-std=c++14"| FileCheck %s -check-prefix=equal_range
// equal_range:  /*1*/ dpct::equal_range(oneapi::dpl::execution::make_device_policy(q_ct1), device_vec.begin(), device_vec.end(), 0);
// equal_range-NEXT: /*2*/  dpct::equal_range(oneapi::dpl::execution::make_device_policy(q_ct1), device_vec.begin(), device_vec.end(), 0);
// equal_range-NEXT: /*3*/  dpct::equal_range(oneapi::dpl::execution::make_device_policy(q_ct1), device_vec.begin(), device_vec.end(), 0, oneapi::dpl::less<int>());
// equal_range-NEXT: /*4*/  dpct::equal_range(oneapi::dpl::execution::make_device_policy(q_ct1), device_vec.begin(), device_vec.end(), 0, oneapi::dpl::less<int>());
// equal_range-NEXT: /*5*/  dpct::equal_range(oneapi::dpl::execution::seq, host_vec.begin(), host_vec.end(), 0);
// equal_range-NEXT: /*6*/  dpct::equal_range(oneapi::dpl::execution::seq, host_vec.begin(), host_vec.end(), 0);
// equal_range-NEXT: /*7*/  dpct::equal_range(oneapi::dpl::execution::seq, host_vec.begin(), host_vec.end(), 0, oneapi::dpl::less<int>());
// equal_range-NEXT: /*8*/  dpct::equal_range(oneapi::dpl::execution::seq, host_vec.begin(), host_vec.end(), 0, oneapi::dpl::less<int>());
// equal_range-NEXT: /*9*/  dpct::equal_range(oneapi::dpl::execution::seq, data, data + N, 0);
// equal_range-NEXT: /*10*/  dpct::equal_range(oneapi::dpl::execution::seq, data, data + N, 0);
// equal_range-NEXT: /*11*/  dpct::equal_range(oneapi::dpl::execution::seq, data, data + N, 0, oneapi::dpl::less<int>());
// equal_range-NEXT: /*12*/  dpct::equal_range(oneapi::dpl::execution::seq, data, data + N, 0, oneapi::dpl::less<int>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::reverse --extra-arg="-std=c++14"| FileCheck %s -check-prefix=reverse
// reverse:   /*1*/ oneapi::dpl::reverse(oneapi::dpl::execution::make_device_policy(q_ct1), device_data.begin(), device_data.end());
// reverse-NEXT:  /*2*/ oneapi::dpl::reverse(oneapi::dpl::execution::seq, host_data.begin(), host_data.end());
// reverse-NEXT:  /*3*/ oneapi::dpl::reverse(oneapi::dpl::execution::seq, data, data + N);
// reverse-NEXT:  /*4*/ oneapi::dpl::reverse(oneapi::dpl::execution::make_device_policy(q_ct1), device_data.begin(), device_data.end());
// reverse-NEXT:  /*5*/ oneapi::dpl::reverse(oneapi::dpl::execution::seq, host_data.begin(), host_data.end());
// reverse-NEXT:  /*6*/ oneapi::dpl::reverse(oneapi::dpl::execution::seq, data, data + N);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::replace_copy --extra-arg="-std=c++14"| FileCheck %s -check-prefix=replace_copy
// replace_copy:  /*1*/ oneapi::dpl::replace_copy(oneapi::dpl::execution::make_device_policy(q_ct1), d_data.begin(), d_data.end(), d_result.begin(), 1, 99);
// replace_copy-NEXT:  /*2*/ oneapi::dpl::replace_copy(oneapi::dpl::execution::seq, h_data.begin(), h_data.end(), h_result.begin(), 1, 99);
// replace_copy-NEXT:  /*3*/ oneapi::dpl::replace_copy(oneapi::dpl::execution::seq, data, data + N, result, 1, 99);
// replace_copy-NEXT:  /*4*/ oneapi::dpl::replace_copy(oneapi::dpl::execution::make_device_policy(q_ct1), d_data.begin(), d_data.end(), d_result.begin(), 1, 99);
// replace_copy-NEXT:  /*5*/ oneapi::dpl::replace_copy(oneapi::dpl::execution::seq, h_data.begin(), h_data.end(), h_result.begin(), 1, 99);
// replace_copy-NEXT:  /*6*/ oneapi::dpl::replace_copy(oneapi::dpl::execution::seq, data, data + N, result, 1, 99);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::mismatch --extra-arg="-std=c++14"| FileCheck %s -check-prefix=mismatch
// mismatch:  /*1*/ oneapi::dpl::mismatch(oneapi::dpl::execution::seq, VA.begin(), VA.end(), VB.begin());
// mismatch-NEXT:  /*2*/ oneapi::dpl::mismatch(oneapi::dpl::execution::seq, VA.begin(), VA.end(), VB.begin());
// mismatch-NEXT:  /*3*/ oneapi::dpl::mismatch(oneapi::dpl::execution::seq, VA.begin(), VA.end(), VB.begin(), oneapi::dpl::equal_to<int>());
// mismatch-NEXT:  /*4*/ oneapi::dpl::mismatch(oneapi::dpl::execution::seq, VA.begin(), VA.end(), VB.begin(), oneapi::dpl::equal_to<int>());
// mismatch-NEXT:  /*5*/ oneapi::dpl::mismatch(oneapi::dpl::execution::make_device_policy(q_ct1), d_VA.begin(), d_VA.end(), d_VB.begin());
// mismatch-NEXT:  /*6*/ oneapi::dpl::mismatch(oneapi::dpl::execution::make_device_policy(q_ct1), d_VA.begin(), d_VA.end(), d_VB.begin());
// mismatch-NEXT:  /*7*/ oneapi::dpl::mismatch(oneapi::dpl::execution::make_device_policy(q_ct1), d_VA.begin(), d_VA.end(), d_VB.begin(), oneapi::dpl::equal_to<int>());
// mismatch-NEXT:  /*8*/ oneapi::dpl::mismatch(oneapi::dpl::execution::make_device_policy(q_ct1), d_VA.begin(), d_VA.end(), d_VB.begin(), oneapi::dpl::equal_to<int>());
// mismatch-NEXT:  /*9*/ oneapi::dpl::mismatch(oneapi::dpl::execution::seq, A, A + N, B);
// mismatch-NEXT:  /*10*/ oneapi::dpl::mismatch(oneapi::dpl::execution::seq, A, A + N, B);
// mismatch-NEXT:  /*11*/ oneapi::dpl::mismatch(oneapi::dpl::execution::seq, A, A + N, B, oneapi::dpl::equal_to<int>());
// mismatch-NEXT:  /*12*/ oneapi::dpl::mismatch(oneapi::dpl::execution::seq, A, A + N, B, oneapi::dpl::equal_to<int>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::find_if_not --extra-arg="-std=c++14"| FileCheck %s -check-prefix=find_if_not
// find_if_not:  /*1*/ oneapi::dpl::find_if_not(oneapi::dpl::execution::seq, data, data + 3, greater_than_four());
// find_if_not-NEXT:  /*2*/ oneapi::dpl::find_if_not(oneapi::dpl::execution::make_device_policy(q_ct1), device_data.begin(), device_data.end(), greater_than_four());
// find_if_not-NEXT:  /*3*/ oneapi::dpl::find_if_not(oneapi::dpl::execution::seq, host_data.begin(), host_data.end(), greater_than_four());
// find_if_not-NEXT:  /*4*/ oneapi::dpl::find_if_not(oneapi::dpl::execution::seq, data, data + 3, greater_than_four());
// find_if_not-NEXT:  /*5*/ oneapi::dpl::find_if_not(oneapi::dpl::execution::make_device_policy(q_ct1), device_data.begin(), device_data.end(), greater_than_four());
// find_if_not-NEXT:  /*6*/ oneapi::dpl::find_if_not(oneapi::dpl::execution::seq, host_data.begin(), host_data.end(), greater_than_four());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::find_if --extra-arg="-std=c++14"| FileCheck %s -check-prefix=find_if
// find_if:  /*1*/ oneapi::dpl::find_if(oneapi::dpl::execution::seq, data, data + 3, greater_than_four());
// find_if-NEXT:  /*2*/ oneapi::dpl::find_if(oneapi::dpl::execution::make_device_policy(q_ct1), device_data.begin(), device_data.end(), greater_than_four());
// find_if-NEXT:  /*3*/ oneapi::dpl::find_if(oneapi::dpl::execution::seq, host_data.begin(), host_data.end(), greater_than_four());
// find_if-NEXT:  /*4*/ oneapi::dpl::find_if(oneapi::dpl::execution::seq, data, data + 3, greater_than_four());
// find_if-NEXT:  /*5*/ oneapi::dpl::find_if(oneapi::dpl::execution::make_device_policy(q_ct1), device_data.begin(), device_data.end(), greater_than_four());
// find_if-NEXT:  /*6*/ oneapi::dpl::find_if(oneapi::dpl::execution::seq, host_data.begin(), host_data.end(), greater_than_four());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::remove --extra-arg="-std=c++14"| FileCheck %s -check-prefix=remove
// remove:  /*1*/ oneapi::dpl::remove(oneapi::dpl::execution::seq, data, data + N, 1);
// remove-NEXT:  /*2*/ oneapi::dpl::remove(oneapi::dpl::execution::seq, host_data.begin(), host_data.begin() + N, 1);
// remove-NEXT:  /*3*/ oneapi::dpl::remove(oneapi::dpl::execution::make_device_policy(q_ct1), device_data.begin(), device_data.begin() + N, 1);
// remove-NEXT:  /*4*/ oneapi::dpl::remove(oneapi::dpl::execution::seq, data, data + N, 1);
// remove-NEXT:  /*5*/ oneapi::dpl::remove(oneapi::dpl::execution::seq, host_data.begin(), host_data.begin() + N, 1);
// remove-NEXT:  /*6*/ oneapi::dpl::remove(oneapi::dpl::execution::make_device_policy(q_ct1), device_data.begin(), device_data.begin() + N, 1);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::stable_partition_copy --extra-arg="-std=c++14"| FileCheck %s -check-prefix=stable_partition_copy
// stable_partition_copy:  /*1*/ dpct::stable_partition_copy(oneapi::dpl::execution::seq, data, data + N, evens, odds, is_even());
// stable_partition_copy-NEXT:  /*2*/ dpct::stable_partition_copy(oneapi::dpl::execution::seq, host_a.begin(), host_a.begin() + N, host_evens.begin(), host_odds.begin(), is_even());
// stable_partition_copy-NEXT:  /*3*/ dpct::stable_partition_copy(oneapi::dpl::execution::make_device_policy(q_ct1), device_a.begin(), device_a.begin() + N, device_evens.begin(), device_odds.begin(), is_even());
// stable_partition_copy-NEXT:  /*4*/ dpct::stable_partition_copy(oneapi::dpl::execution::seq, data, data + N, evens, odds, is_even());
// stable_partition_copy-NEXT:  /*5*/ dpct::stable_partition_copy(oneapi::dpl::execution::seq, host_a.begin(), host_a.begin() + N, host_evens.begin(), host_odds.begin(), is_even());
// stable_partition_copy-NEXT:  /*6*/ dpct::stable_partition_copy(oneapi::dpl::execution::make_device_policy(q_ct1), device_a.begin(), device_a.begin() + N, device_evens.begin(), device_odds.begin(), is_even());
// stable_partition_copy-NEXT:  /*7*/ dpct::stable_partition_copy(oneapi::dpl::execution::seq, data, data + N, S, evens, odds, is_even());
// stable_partition_copy-NEXT:  /*8*/ dpct::stable_partition_copy(oneapi::dpl::execution::seq, host_a.begin(), host_a.begin() + N, host_S.begin(), host_evens.begin(), host_odds.begin(), is_even());
// stable_partition_copy-NEXT:  /*9*/ dpct::stable_partition_copy(oneapi::dpl::execution::make_device_policy(q_ct1), device_a.begin(), device_a.begin() + N, device_S.begin(), device_evens.begin(), device_odds.begin(), is_even());
// stable_partition_copy-NEXT:  /*10*/ dpct::stable_partition_copy(oneapi::dpl::execution::seq, data, data + N, S, evens, odds, is_even());
// stable_partition_copy-NEXT:  /*11*/ dpct::stable_partition_copy(oneapi::dpl::execution::seq, host_a.begin(), host_a.begin() + N, host_S.begin(), host_evens.begin(), host_odds.begin(), is_even());
// stable_partition_copy-NEXT:  /*12*/ dpct::stable_partition_copy(oneapi::dpl::execution::make_device_policy(q_ct1), device_a.begin(), device_a.begin() + N, device_S.begin(), device_evens.begin(), device_odds.begin(), is_even());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::stable_partition --extra-arg="-std=c++14"| FileCheck %s -check-prefix=stable_partition
// stable_partition:  /*1*/ oneapi::dpl::stable_partition(oneapi::dpl::execution::seq, data, data + N, is_even());
// stable_partition-NEXT:  /*2*/ oneapi::dpl::stable_partition(oneapi::dpl::execution::seq, host_data.begin(), host_data.begin() + N, is_even());
// stable_partition-NEXT:  /*3*/ oneapi::dpl::stable_partition(oneapi::dpl::execution::make_device_policy(q_ct1), device_data.begin(), device_data.begin() + N, is_even());
// stable_partition-NEXT:  /*4*/ oneapi::dpl::stable_partition(oneapi::dpl::execution::seq, data, data + N, is_even());
// stable_partition-NEXT:  /*5*/ oneapi::dpl::stable_partition(oneapi::dpl::execution::seq, host_data.begin(), host_data.begin() + N, is_even());
// stable_partition-NEXT:  /*6*/ oneapi::dpl::stable_partition(oneapi::dpl::execution::make_device_policy(q_ct1), device_data.begin(), device_data.begin() + N, is_even());
// stable_partition-NEXT:  /*7*/ dpct::stable_partition(oneapi::dpl::execution::seq, data, data + N, S, is_even());
// stable_partition-NEXT:  /*8*/ dpct::stable_partition(oneapi::dpl::execution::seq, host_data.begin(), host_data.begin() + N, host_S.begin(), is_even());
// stable_partition-NEXT:  /*9*/ dpct::stable_partition(oneapi::dpl::execution::make_device_policy(q_ct1), device_data.begin(), device_data.begin() + N, device_s.begin(), is_even());
// stable_partition-NEXT:  /*10*/ dpct::stable_partition(oneapi::dpl::execution::seq, data, data + N, S, is_even());
// stable_partition-NEXT:  /*11*/ dpct::stable_partition(oneapi::dpl::execution::seq, host_data.begin(), host_data.begin() + N, host_S.begin(), is_even());
// stable_partition-NEXT:  /*12*/ dpct::stable_partition(oneapi::dpl::execution::make_device_policy(q_ct1), device_data.begin(), device_data.begin() + N, device_s.begin(), is_even());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::scatter_if --extra-arg="-std=c++14"| FileCheck %s -check-prefix=scatter_if
// scatter_if:  /*1*/ dpct::scatter_if(oneapi::dpl::execution::seq, V, V + 8, M, S, D);
// scatter_if-NEXT:  /*2*/ dpct::scatter_if(oneapi::dpl::execution::seq, V, V + 8, M, S, D);
// scatter_if-NEXT:  /*3*/ dpct::scatter_if(oneapi::dpl::execution::seq, V, V + 8, M, S, D, pred);
// scatter_if-NEXT:  /*4*/ dpct::scatter_if(oneapi::dpl::execution::seq, V, V + 8, M, S, D, pred);
// scatter_if-NEXT:  /*5*/  dpct::scatter_if(oneapi::dpl::execution::make_device_policy(q_ct1), d_V.begin(), d_V.end(), d_M.begin(), d_S.begin(), d_D.begin());
// scatter_if-NEXT:  /*6*/  dpct::scatter_if(oneapi::dpl::execution::make_device_policy(q_ct1), d_V.begin(), d_V.end(), d_M.begin(), d_S.begin(), d_D.begin());
// scatter_if-NEXT:  /*7*/  dpct::scatter_if(oneapi::dpl::execution::make_device_policy(q_ct1), d_V.begin(), d_V.end(), d_M.begin(), d_S.begin(), d_D.begin(), pred);
// scatter_if-NEXT:  /*8*/  dpct::scatter_if(oneapi::dpl::execution::make_device_policy(q_ct1), d_V.begin(), d_V.end(), d_M.begin(), d_S.begin(), d_D.begin(), pred);
// scatter_if-NEXT:  /*9*/  dpct::scatter_if(oneapi::dpl::execution::seq, h_V.begin(), h_V.end(), h_M.begin(), h_S.begin(), h_D.begin());
// scatter_if-NEXT:  /*10*/  dpct::scatter_if(oneapi::dpl::execution::seq, h_V.begin(), h_V.end(), h_M.begin(), h_S.begin(), h_D.begin());
// scatter_if-NEXT:  /*11*/  dpct::scatter_if(oneapi::dpl::execution::seq, h_V.begin(), h_V.end(), h_M.begin(), h_S.begin(), h_D.begin(), pred);
// scatter_if-NEXT:  /*12*/  dpct::scatter_if(oneapi::dpl::execution::seq, h_V.begin(), h_V.end(), h_M.begin(), h_S.begin(), h_D.begin(), pred);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::reverse_copy --extra-arg="-std=c++14"| FileCheck %s -check-prefix=reverse_copy
// reverse_copy:  /*1*/  oneapi::dpl::reverse_copy(oneapi::dpl::execution::make_device_policy(q_ct1), device_data.begin(), device_data.end(), device_result.begin());
// reverse_copy-NEXT:  /*2*/  oneapi::dpl::reverse_copy(oneapi::dpl::execution::seq, host_data.begin(), host_data.end(), host_result.begin());
// reverse_copy-NEXT:  /*3*/  oneapi::dpl::reverse_copy(oneapi::dpl::execution::seq, data, data + N, result);
// reverse_copy-NEXT:  /*4*/  oneapi::dpl::reverse_copy(oneapi::dpl::execution::make_device_policy(q_ct1), device_data.begin(), device_data.end(), device_result.begin());
// reverse_copy-NEXT:  /*5*/  oneapi::dpl::reverse_copy(oneapi::dpl::execution::seq, host_data.begin(), host_data.end(), host_result.begin());
// reverse_copy-NEXT:  /*6*/  oneapi::dpl::reverse_copy(oneapi::dpl::execution::seq, data, data + N, result);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::minmax_element --extra-arg="-std=c++14"| FileCheck %s -check-prefix=minmax_element
// minmax_element:  /*1*/ oneapi::dpl::minmax_element(oneapi::dpl::execution::seq, h_values.begin(), h_values.end());
// minmax_element-NEXT:  /*2*/ oneapi::dpl::minmax_element(oneapi::dpl::execution::seq, h_values.begin(), h_values.end());
// minmax_element-NEXT:  /*3*/ oneapi::dpl::minmax_element(oneapi::dpl::execution::seq, h_values.begin(), h_values.begin() + 4, compare_key_value());
// minmax_element-NEXT:  /*4*/ oneapi::dpl::minmax_element(oneapi::dpl::execution::seq, h_values.begin(), h_values.begin() + 4, compare_key_value());
// minmax_element-NEXT:  /*5*/ oneapi::dpl::minmax_element(oneapi::dpl::execution::make_device_policy(q_ct1), d_values.begin(), d_values.end());
// minmax_element-NEXT:  /*6*/ oneapi::dpl::minmax_element(oneapi::dpl::execution::make_device_policy(q_ct1), d_values.begin(), d_values.end());
// minmax_element-NEXT:  /*7*/ oneapi::dpl::minmax_element(oneapi::dpl::execution::make_device_policy(q_ct1), d_values.begin(), d_values.end(), compare_key_value());
// minmax_element-NEXT:  /*8*/ oneapi::dpl::minmax_element(oneapi::dpl::execution::make_device_policy(q_ct1), d_values.begin(), d_values.end(), compare_key_value());
// minmax_element-NEXT:  /*9*/ oneapi::dpl::minmax_element(oneapi::dpl::execution::seq, data, data + N);
// minmax_element-NEXT:  /*10*/ oneapi::dpl::minmax_element(oneapi::dpl::execution::seq, data, data + N);
// minmax_element-NEXT:  /*11*/ oneapi::dpl::minmax_element(oneapi::dpl::execution::seq, data, data + N, compare_key_value());
// minmax_element-NEXT:  /*12*/ oneapi::dpl::minmax_element(oneapi::dpl::execution::seq, data, data + N, compare_key_value());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::unique_by_key --extra-arg="-std=c++14"| FileCheck %s -check-prefix=unique_by_key
// unique_by_key:  /*1*/ dpct::unique(oneapi::dpl::execution::seq, h_keys.begin(), h_keys.end(), h_values.begin());
// unique_by_key-NEXT:  /*2*/ dpct::unique(oneapi::dpl::execution::seq, h_keys.begin(), h_keys.end(), h_values.begin());
// unique_by_key-NEXT:   /*3*/ dpct::unique(oneapi::dpl::execution::seq, h_keys.begin(), h_keys.end(), h_values.begin(), binary_pred);
// unique_by_key-NEXT:   /*4*/ dpct::unique(oneapi::dpl::execution::seq, h_keys.begin(), h_keys.end(), h_values.begin(), binary_pred);
// unique_by_key-NEXT:   /*5*/ dpct::unique(oneapi::dpl::execution::make_device_policy(q_ct1), d_keys.begin(), d_keys.end(), d_values.begin());
// unique_by_key-NEXT:   /*6*/ dpct::unique(oneapi::dpl::execution::make_device_policy(q_ct1), d_keys.begin(), d_keys.end(), d_values.begin());
// unique_by_key-NEXT:   /*7*/ dpct::unique(oneapi::dpl::execution::make_device_policy(q_ct1), d_keys.begin(), d_keys.end(), d_values.begin(), binary_pred);
// unique_by_key-NEXT:   /*8*/ dpct::unique(oneapi::dpl::execution::make_device_policy(q_ct1), d_keys.begin(), d_keys.end(), d_values.begin(), binary_pred);
// unique_by_key-NEXT:   /*9*/ dpct::unique(oneapi::dpl::execution::seq, A, A + N, B);
// unique_by_key-NEXT:   /*10*/ dpct::unique(oneapi::dpl::execution::seq, A, A + N, B);
// unique_by_key-NEXT:   /*11*/ dpct::unique(oneapi::dpl::execution::seq, A, A + N, B, binary_pred);
// unique_by_key-NEXT:   /*12*/ dpct::unique(oneapi::dpl::execution::seq, A, A + N, B, binary_pred);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::is_sorted --extra-arg="-std=c++14"| FileCheck %s -check-prefix=is_sorted
// is_sorted:  /*1*/ oneapi::dpl::is_sorted(oneapi::dpl::execution::seq, h_v.begin(), h_v.end());
// is_sorted-NEXT:  /*2*/ oneapi::dpl::is_sorted(oneapi::dpl::execution::seq, h_v.begin(), h_v.end());
// is_sorted-NEXT:  /*3*/ oneapi::dpl::is_sorted(oneapi::dpl::execution::seq, h_v.begin(), h_v.end(), comp);
// is_sorted-NEXT:  /*4*/ oneapi::dpl::is_sorted(oneapi::dpl::execution::seq, h_v.begin(), h_v.end(), comp);
// is_sorted-NEXT:  /*5*/ oneapi::dpl::is_sorted(oneapi::dpl::execution::seq, h_v.begin(), h_v.end(), comp);
// is_sorted-NEXT:  /*6*/ oneapi::dpl::is_sorted(oneapi::dpl::execution::make_device_policy(q_ct1), d_v.begin(), d_v.end());
// is_sorted-NEXT:  /*7*/ oneapi::dpl::is_sorted(oneapi::dpl::execution::make_device_policy(q_ct1), d_v.begin(), d_v.end(), comp);
// is_sorted-NEXT:  /*8*/ oneapi::dpl::is_sorted(oneapi::dpl::execution::make_device_policy(q_ct1), d_v.begin(), d_v.end(), comp);
// is_sorted-NEXT:  /*9*/ oneapi::dpl::is_sorted(oneapi::dpl::execution::seq, datas, datas + N);
// is_sorted-NEXT:  /*10*/ oneapi::dpl::is_sorted(oneapi::dpl::execution::seq, datas, datas + N);
// is_sorted-NEXT:  /*11*/ oneapi::dpl::is_sorted(oneapi::dpl::execution::seq, datas, datas + N, comp);
// is_sorted-NEXT:  /*12*/ oneapi::dpl::is_sorted(oneapi::dpl::execution::seq, datas, datas + N, comp);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::partition --extra-arg="-std=c++14"| FileCheck %s -check-prefix=partition
// partition:  /*1*/ oneapi::dpl::partition(oneapi::dpl::execution::seq, h_v.begin(), h_v.end(), is_even());
// partition-NEXT:  /*2*/ oneapi::dpl::partition(oneapi::dpl::execution::seq, h_v.begin(), h_v.end(), is_even());
// partition-NEXT:  /*3*/ dpct::partition(oneapi::dpl::execution::seq, h_vdata.begin(), h_vdata.end(), h_vstencil.begin(), is_even());
// partition-NEXT:  /*4*/ dpct::partition(oneapi::dpl::execution::seq, h_vdata.begin(), h_vdata.end(), h_vstencil.begin(), is_even());
// partition-NEXT:  /*5*/ oneapi::dpl::partition(oneapi::dpl::execution::make_device_policy(q_ct1), d_v.begin(), d_v.end(), is_even());
// partition-NEXT:  /*6*/ oneapi::dpl::partition(oneapi::dpl::execution::make_device_policy(q_ct1), d_v.begin(), d_v.end(), is_even());
// partition-NEXT:  /*7*/ dpct::partition(oneapi::dpl::execution::make_device_policy(q_ct1), d_vdata.begin(), d_vdata.end(), d_vstencil.begin(), is_even());
// partition-NEXT:  /*8*/ dpct::partition(oneapi::dpl::execution::make_device_policy(q_ct1), d_vdata.begin(), d_vdata.end(), d_vstencil.begin(), is_even());
// partition-NEXT:  /*9*/ oneapi::dpl::partition(oneapi::dpl::execution::seq, datas, datas + N, is_even());
// partition-NEXT:  /*10*/ oneapi::dpl::partition(oneapi::dpl::execution::seq, datas, datas + N, is_even());
// partition-NEXT:  /*11*/ dpct::partition(oneapi::dpl::execution::seq, datas, datas + N, stencil, is_even());
// partition-NEXT:  /*12*/ dpct::partition(oneapi::dpl::execution::seq, datas, datas + N, stencil, is_even());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::unique_copy --extra-arg="-std=c++14"| FileCheck %s -check-prefix=unique_copy
// unique_copy:  /*1*/ oneapi::dpl::unique_copy(oneapi::dpl::execution::seq, h_V.begin(), h_V.end(), h_result.begin());
// unique_copy-NEXT:  /*2*/ oneapi::dpl::unique_copy(oneapi::dpl::execution::seq, h_V.begin(), h_V.end(), h_result.begin());
// unique_copy-NEXT:  /*3*/ oneapi::dpl::unique_copy(oneapi::dpl::execution::seq, h_V.begin(), h_V.end(), h_result.begin(), oneapi::dpl::equal_to<int>());
// unique_copy-NEXT:  /*4*/ oneapi::dpl::unique_copy(oneapi::dpl::execution::seq, h_V.begin(), h_V.end(), h_result.begin(), oneapi::dpl::equal_to<int>());
// unique_copy-NEXT:  /*5*/ oneapi::dpl::unique_copy(oneapi::dpl::execution::make_device_policy(q_ct1), d_V.begin(), d_V.end(), d_result.begin());
// unique_copy-NEXT:  /*6*/ oneapi::dpl::unique_copy(oneapi::dpl::execution::make_device_policy(q_ct1), d_V.begin(), d_V.end(), d_result.begin());
// unique_copy-NEXT:  /*7*/ oneapi::dpl::unique_copy(oneapi::dpl::execution::make_device_policy(q_ct1), d_V.begin(), d_V.end(), d_result.begin(), oneapi::dpl::equal_to<int>());
// unique_copy-NEXT:  /*8*/ oneapi::dpl::unique_copy(oneapi::dpl::execution::make_device_policy(q_ct1), d_V.begin(), d_V.end(), d_result.begin(), oneapi::dpl::equal_to<int>());
// unique_copy-NEXT:  /*9*/ oneapi::dpl::unique_copy(oneapi::dpl::execution::seq, A, A + N, B);
// unique_copy-NEXT:  /*10*/ oneapi::dpl::unique_copy(oneapi::dpl::execution::seq, A, A + N, B);
// unique_copy-NEXT:  /*11*/ oneapi::dpl::unique_copy(oneapi::dpl::execution::seq, A, A + N, B, oneapi::dpl::equal_to<int>());
// unique_copy-NEXT:  /*12*/ oneapi::dpl::unique_copy(oneapi::dpl::execution::seq, A, A + N, B, oneapi::dpl::equal_to<int>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::stable_sort --extra-arg="-std=c++14"| FileCheck %s -check-prefix=stable_sort
// stable_sort:  /*1*/ oneapi::dpl::stable_sort(oneapi::dpl::execution::seq, h_v.begin(), h_v.end());
// stable_sort-NEXT:  /*2*/ oneapi::dpl::stable_sort(oneapi::dpl::execution::seq, h_v.begin(), h_v.end());
// stable_sort-NEXT:  /*3*/ oneapi::dpl::stable_sort(oneapi::dpl::execution::seq, h_v.begin(), h_v.end(), std::greater<int>());
// stable_sort-NEXT:  /*4*/ oneapi::dpl::stable_sort(oneapi::dpl::execution::seq, h_v.begin(), h_v.end(), std::greater<int>());
// stable_sort-NEXT:  /*5*/ oneapi::dpl::stable_sort(oneapi::dpl::execution::make_device_policy(q_ct1), d_v.begin(), d_v.end());
// stable_sort-NEXT:  /*6*/ oneapi::dpl::stable_sort(oneapi::dpl::execution::make_device_policy(q_ct1), d_v.begin(), d_v.end());
// stable_sort-NEXT:  /*7*/ oneapi::dpl::stable_sort(oneapi::dpl::execution::make_device_policy(q_ct1), d_v.begin(), d_v.end(), std::greater<int>());
// stable_sort-NEXT:  /*8*/ oneapi::dpl::stable_sort(oneapi::dpl::execution::make_device_policy(q_ct1), d_v.begin(), d_v.end(), std::greater<int>());
// stable_sort-NEXT:  /*9*/ oneapi::dpl::stable_sort(oneapi::dpl::execution::seq, datas, datas + N);
// stable_sort-NEXT:  /*10*/ oneapi::dpl::stable_sort(oneapi::dpl::execution::seq, datas, datas + N);
// stable_sort-NEXT:  /*11*/ oneapi::dpl::stable_sort(oneapi::dpl::execution::seq, datas, datas + N, std::greater<int>());
// stable_sort-NEXT:  /*12*/ oneapi::dpl::stable_sort(oneapi::dpl::execution::seq, datas, datas + N, std::greater<int>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::set_difference_by_key --extra-arg="-std=c++14"| FileCheck %s -check-prefix=set_difference_by_key
// set_difference_by_key:  /*1*/ dpct::set_difference(oneapi::dpl::execution::seq, h_VAkey.begin(), h_VAkey.end(), h_VBkey.begin(), h_VBkey.end(), h_VAvalue.begin(), h_VBvalue.begin(), h_VCkey.begin(), h_VCvalue.begin());
// set_difference_by_key-NEXT:  /*2*/ dpct::set_difference(oneapi::dpl::execution::seq, h_VAkey.begin(), h_VAkey.end(), h_VBkey.begin(), h_VBkey.end(), h_VAvalue.begin(), h_VBvalue.begin(), h_VCkey.begin(), h_VCvalue.begin());
// set_difference_by_key-NEXT:  /*3*/ dpct::set_difference(oneapi::dpl::execution::seq, h_VAkey.begin(), h_VAkey.end(), h_VBkey.begin(), h_VBkey.end(), h_VAvalue.begin(), h_VBvalue.begin(), h_VCkey.begin(), h_VCvalue.begin(), std::greater<int>());
// set_difference_by_key-NEXT:  /*4*/ dpct::set_difference(oneapi::dpl::execution::seq, h_VAkey.begin(), h_VAkey.end(), h_VBkey.begin(), h_VBkey.end(), h_VAvalue.begin(), h_VBvalue.begin(), h_VCkey.begin(), h_VCvalue.begin(), std::greater<int>());
// set_difference_by_key-NEXT:  /*5*/ dpct::set_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_VAkey.begin(), d_VAkey.end(), d_VBkey.begin(), d_VBkey.end(), d_VAvalue.begin(), d_VBvalue.begin(), d_VCkey.begin(), d_VCvalue.begin());
// set_difference_by_key-NEXT:  /*6*/ dpct::set_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_VAkey.begin(), d_VAkey.end(), d_VBkey.begin(), d_VBkey.end(), d_VAvalue.begin(), d_VBvalue.begin(), d_VCkey.begin(), d_VCvalue.begin());
// set_difference_by_key-NEXT:  /*7*/ dpct::set_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_VAkey.begin(), d_VAkey.end(), d_VBkey.begin(), d_VBkey.end(), d_VAvalue.begin(), d_VBvalue.begin(), d_VCkey.begin(), d_VCvalue.begin(), std::greater<int>());
// set_difference_by_key-NEXT:  /*8*/ dpct::set_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_VAkey.begin(), d_VAkey.end(), d_VBkey.begin(), d_VBkey.end(), d_VAvalue.begin(), d_VBvalue.begin(), d_VCkey.begin(), d_VCvalue.begin(), std::greater<int>());
// set_difference_by_key-NEXT:  /*9*/ dpct::set_difference(oneapi::dpl::execution::seq, Akey, Akey + N, Bkey, Bkey + M, Avalue, Bvalue, Ckey, Cvalue);
// set_difference_by_key-NEXT:  /*10*/ dpct::set_difference(oneapi::dpl::execution::seq, Akey, Akey + N, Bkey, Bkey + M, Avalue, Bvalue, Ckey, Cvalue);
// set_difference_by_key-NEXT:  /*11*/ dpct::set_difference(oneapi::dpl::execution::seq, Akey, Akey + N, Bkey, Bkey + M, Avalue, Bvalue, Ckey, Cvalue, std::greater<int>());
// set_difference_by_key-NEXT:  /*12*/ dpct::set_difference(oneapi::dpl::execution::seq, Akey, Akey + N, Bkey, Bkey + M, Avalue, Bvalue, Ckey, Cvalue, std::greater<int>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::set_difference --extra-arg="-std=c++14"| FileCheck %s -check-prefix=set_difference
// set_difference:  /*1*/ oneapi::dpl::set_difference(oneapi::dpl::execution::seq, h_VA.begin(), h_VA.end(), h_VB.begin(), h_VB.end(), h_VC.begin());
// set_difference-NEXT:  /*2*/ oneapi::dpl::set_difference(oneapi::dpl::execution::seq, h_VA.begin(), h_VA.end(), h_VB.begin(), h_VB.end(), h_VC.begin());
// set_difference-NEXT:  /*3*/ oneapi::dpl::set_difference(oneapi::dpl::execution::seq, h_VA.begin(), h_VA.end(), h_VB.begin(), h_VB.end(), h_VC.begin(), std::greater<int>());
// set_difference-NEXT:  /*4*/ oneapi::dpl::set_difference(oneapi::dpl::execution::seq, h_VA.begin(), h_VA.end(), h_VB.begin(), h_VB.end(), h_VC.begin(), std::greater<int>());
// set_difference-NEXT:  /*5*/ oneapi::dpl::set_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_VA.begin(), d_VA.end(), d_VB.begin(), d_VB.end(), d_VC.begin());
// set_difference-NEXT:  /*6*/ oneapi::dpl::set_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_VA.begin(), d_VA.end(), d_VB.begin(), d_VB.end(), d_VC.begin());
// set_difference-NEXT:  /*7*/ oneapi::dpl::set_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_VA.begin(), d_VA.end(), d_VB.begin(), d_VB.end(), d_VC.begin(), std::greater<int>());
// set_difference-NEXT:  /*8*/ oneapi::dpl::set_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_VA.begin(), d_VA.end(), d_VB.begin(), d_VB.end(), d_VC.begin(), std::greater<int>());
// set_difference-NEXT:  /*9*/ oneapi::dpl::set_difference(oneapi::dpl::execution::seq, A, A + N, B, B + M, C);
// set_difference-NEXT:  /*10*/ oneapi::dpl::set_difference(oneapi::dpl::execution::seq, A, A + N, B, B + M, C);
// set_difference-NEXT:  /*11*/ oneapi::dpl::set_difference(oneapi::dpl::execution::seq, A, A + N, B, B + M, C, std::greater<int>());
// set_difference-NEXT:  /*12*/ oneapi::dpl::set_difference(oneapi::dpl::execution::seq, A, A + N, B, B + M, C, std::greater<int>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::for_each_n --extra-arg="-std=c++14"| FileCheck %s -check-prefix=for_each_n
// for_each_n:  /*1*/ oneapi::dpl::for_each_n(oneapi::dpl::execution::seq, h_V.begin(), h_V.size(), add_functor());
// for_each_n-NEXT:  /*2*/ oneapi::dpl::for_each_n(oneapi::dpl::execution::seq, h_V.begin(), h_V.size(), add_functor());
// for_each_n-NEXT:  /*3*/ oneapi::dpl::for_each_n(oneapi::dpl::execution::make_device_policy(q_ct1), d_V.begin(), d_V.size(), add_functor());
// for_each_n-NEXT:  /*4*/ oneapi::dpl::for_each_n(oneapi::dpl::execution::seq, d_V.begin(), d_V.size(), add_functor());
// for_each_n-NEXT:  /*5*/ oneapi::dpl::for_each_n(oneapi::dpl::execution::seq, A, N, add_functor());
// for_each_n-NEXT:  /*6*/ oneapi::dpl::for_each_n(oneapi::dpl::execution::seq, A, N, add_functor());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::tabulate --extra-arg="-std=c++14"| FileCheck %s -check-prefix=tabulate
// tabulate:  /*1*/ dpct::for_each_index(oneapi::dpl::execution::seq, h_V.begin(), h_V.end(), std::negate<int>());
// tabulate-NEXT:  /*2*/ dpct::for_each_index(oneapi::dpl::execution::seq, h_V.begin(), h_V.end(), std::negate<int>());
// tabulate-NEXT:  /*3*/ dpct::for_each_index(oneapi::dpl::execution::make_device_policy(q_ct1), d_V.begin(), d_V.end(), std::negate<int>());
// tabulate-NEXT:  /*4*/ dpct::for_each_index(oneapi::dpl::execution::make_device_policy(q_ct1), d_V.begin(), d_V.end(), std::negate<int>());
// tabulate-NEXT:  /*5*/ dpct::for_each_index(oneapi::dpl::execution::seq, A, A + N, std::negate<int>());
// tabulate-NEXT:  /*6*/ dpct::for_each_index(oneapi::dpl::execution::seq, A, A + N, std::negate<int>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::remove_copy --extra-arg="-std=c++14"| FileCheck %s -check-prefix=remove_copy
// remove_copy:  /*1*/ oneapi::dpl::remove_copy(oneapi::dpl::execution::seq, h_V.begin(), h_V.end(), h_result.begin(), 0);
// remove_copy-NEXT:  /*2*/ oneapi::dpl::remove_copy(oneapi::dpl::execution::seq, h_V.begin(), h_V.end(), h_result.begin(), 0);
// remove_copy-NEXT:  /*3*/ oneapi::dpl::remove_copy(oneapi::dpl::execution::make_device_policy(q_ct1), d_V.begin(), d_V.end(), d_result.begin(), 0);
// remove_copy-NEXT:  /*4*/ oneapi::dpl::remove_copy(oneapi::dpl::execution::make_device_policy(q_ct1), d_V.begin(), d_V.end(), d_result.begin(), 0);
// remove_copy-NEXT:  /*5*/ oneapi::dpl::remove_copy(oneapi::dpl::execution::seq, V, V + N, result, 0);
// remove_copy-NEXT:  /*6*/ oneapi::dpl::remove_copy(oneapi::dpl::execution::seq, V, V + N, result, 0);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::transform_exclusive_scan --extra-arg="-std=c++14"| FileCheck %s -check-prefix=transform_exclusive_scan
// transform_exclusive_scan:  /*1*/ oneapi::dpl::transform_exclusive_scan(oneapi::dpl::execution::seq, h_V.begin(), h_V.end(), h_V.begin(), 4, binary_op, unary_op);
// transform_exclusive_scan-NEXT:  /*2*/ oneapi::dpl::transform_exclusive_scan(oneapi::dpl::execution::seq, h_V.begin(), h_V.end(), h_V.begin(), 4, binary_op, unary_op);
// transform_exclusive_scan-NEXT:  /*3*/ oneapi::dpl::transform_exclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), d_V.begin(), d_V.end(), d_V.begin(), 4, binary_op, unary_op);
// transform_exclusive_scan-NEXT:  /*4*/ oneapi::dpl::transform_exclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), d_V.begin(), d_V.end(), d_V.begin(), 4, binary_op, unary_op);
// transform_exclusive_scan-NEXT:  /*5*/ oneapi::dpl::transform_exclusive_scan(oneapi::dpl::execution::seq, A, A + N, A, 4, binary_op, unary_op);
// transform_exclusive_scan-NEXT:  /*6*/ oneapi::dpl::transform_exclusive_scan(oneapi::dpl::execution::seq, A, A + N, A, 4, binary_op, unary_op);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::set_intersection_by_key --extra-arg="-std=c++14"| FileCheck %s -check-prefix=set_intersection_by_key
// set_intersection_by_key:  /*1*/ dpct::set_intersection(oneapi::dpl::execution::seq, h_VAkey.begin(), h_VAkey.end(), h_VBkey.begin(), h_VBkey.end(), h_VAvalue.begin(), h_VCkey.begin(), h_VCvalue.begin());
// set_intersection_by_key-NEXT:  /*2*/ dpct::set_intersection(oneapi::dpl::execution::seq, h_VAkey.begin(), h_VAkey.end(), h_VBkey.begin(), h_VBkey.end(), h_VAvalue.begin(), h_VCkey.begin(), h_VCvalue.begin());
// set_intersection_by_key-NEXT:  /*3*/ dpct::set_intersection(oneapi::dpl::execution::seq, h_VAkey.begin(), h_VAkey.end(), h_VBkey.begin(), h_VBkey.end(), h_VAvalue.begin(), h_VCkey.begin(), h_VCvalue.begin(), std::greater<int>());
// set_intersection_by_key-NEXT:  /*4*/ dpct::set_intersection(oneapi::dpl::execution::seq, h_VAkey.begin(), h_VAkey.end(), h_VBkey.begin(), h_VBkey.end(), h_VAvalue.begin(), h_VCkey.begin(), h_VCvalue.begin(), std::greater<int>());
// set_intersection_by_key-NEXT:  /*5*/ dpct::set_intersection(oneapi::dpl::execution::make_device_policy(q_ct1), d_VAkey.begin(), d_VAkey.end(), d_VBkey.begin(), d_VBkey.end(), d_VAvalue.begin(), d_VCkey.begin(), d_VCvalue.begin());
// set_intersection_by_key-NEXT:  /*6*/ dpct::set_intersection(oneapi::dpl::execution::make_device_policy(q_ct1), d_VAkey.begin(), d_VAkey.end(), d_VBkey.begin(), d_VBkey.end(), d_VAvalue.begin(), d_VCkey.begin(), d_VCvalue.begin());
// set_intersection_by_key-NEXT:  /*7*/ dpct::set_intersection(oneapi::dpl::execution::make_device_policy(q_ct1), d_VAkey.begin(), d_VAkey.end(), d_VBkey.begin(), d_VBkey.end(), d_VAvalue.begin(), d_VCkey.begin(), d_VCvalue.begin(), std::greater<int>());
// set_intersection_by_key-NEXT:  /*8*/ dpct::set_intersection(oneapi::dpl::execution::make_device_policy(q_ct1), d_VAkey.begin(), d_VAkey.end(), d_VBkey.begin(), d_VBkey.end(), d_VAvalue.begin(), d_VCkey.begin(), d_VCvalue.begin(), std::greater<int>());
// set_intersection_by_key-NEXT:  /*9*/ dpct::set_intersection(oneapi::dpl::execution::seq, Akey, Akey + N, Bkey, Bkey + M, Avalue, Ckey, Cvalue);
// set_intersection_by_key-NEXT:  /*10*/ dpct::set_intersection(oneapi::dpl::execution::seq, Akey, Akey + N, Bkey, Bkey + M, Avalue, Ckey, Cvalue);
// set_intersection_by_key-NEXT:  /*11*/ dpct::set_intersection(oneapi::dpl::execution::seq, Akey, Akey + N, Bkey, Bkey + M, Avalue, Ckey, Cvalue, std::greater<int>());
// set_intersection_by_key-NEXT:  /*12*/ dpct::set_intersection(oneapi::dpl::execution::seq, Akey, Akey + N, Bkey, Bkey + M, Avalue, Ckey, Cvalue, std::greater<int>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::raw_reference_cast --extra-arg="-std=c++14"| FileCheck %s -check-prefix=raw_reference_cast
// raw_reference_cast:  /*1*/ int &ref1 = dpct::get_raw_reference(d_vec[0]);
// raw_reference_cast-NEXT:  /*2*/ int &ref2 = dpct::get_raw_reference(ref_const);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::partition_copy --extra-arg="-std=c++14"| FileCheck %s -check-prefix=partition_copy
// partition_copy:  /*1*/ oneapi::dpl::partition_copy(oneapi::dpl::execution::seq, data, data + N, evens, odds, is_even());
// partition_copy-NEXT:  /*2*/ oneapi::dpl::partition_copy(oneapi::dpl::execution::seq, host_a.begin(), host_a.begin() + N, host_evens.begin(), host_odds.begin(), is_even());
// partition_copy-NEXT:  /*3*/ oneapi::dpl::partition_copy(oneapi::dpl::execution::make_device_policy(q_ct1), device_a.begin(), device_a.begin() + N, device_evens.begin(), device_odds.begin(), is_even());
// partition_copy-NEXT:  /*4*/ oneapi::dpl::partition_copy(oneapi::dpl::execution::seq, data, data + N, evens, odds, is_even());
// partition_copy-NEXT:  /*5*/ oneapi::dpl::partition_copy(oneapi::dpl::execution::seq, host_a.begin(), host_a.begin() + N, host_evens.begin(), host_odds.begin(), is_even());
// partition_copy-NEXT:  /*6*/ oneapi::dpl::partition_copy(oneapi::dpl::execution::make_device_policy(q_ct1), device_a.begin(), device_a.begin() + N, device_evens.begin(), device_odds.begin(), is_even());
// partition_copy-NEXT:  /*7*/ dpct::partition_copy(oneapi::dpl::execution::seq, data, data + N, S, evens, odds, is_even());
// partition_copy-NEXT:  /*8*/ dpct::partition_copy(oneapi::dpl::execution::seq, host_a.begin(), host_a.begin() + N, host_S.begin(), host_evens.begin(), host_odds.begin(), is_even());
// partition_copy-NEXT:  /*9*/ dpct::partition_copy(oneapi::dpl::execution::make_device_policy(q_ct1), device_a.begin(), device_a.begin() + N, device_S.begin(), device_evens.begin(), device_odds.begin(), is_even());
// partition_copy-NEXT:  /*10*/ dpct::partition_copy(oneapi::dpl::execution::seq, data, data + N, S, evens, odds, is_even());
// partition_copy-NEXT:  /*11*/ dpct::partition_copy(oneapi::dpl::execution::seq, host_a.begin(), host_a.begin() + N, host_S.begin(), host_evens.begin(), host_odds.begin(), is_even());
// partition_copy-NEXT:  /*12*/ dpct::partition_copy(oneapi::dpl::execution::make_device_policy(q_ct1), device_a.begin(), device_a.begin() + N, device_S.begin(), device_evens.begin(), device_odds.begin(), is_even());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::swap --extra-arg="-std=c++14"| FileCheck %s -check-prefix=swap_api
// swap_api: std::swap(x, y)

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::device_pointer_cast --extra-arg="-std=c++14"| FileCheck %s -check-prefix=device_pointer_cast
// device_pointer_cast: dpct::device_pointer<int> begin = dpct::get_device_pointer(&data[0]);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::raw_pointer_cast --extra-arg="-std=c++14"| FileCheck %s -check-prefix=raw_pointer_cast
// raw_pointer_cast: auto min_costs_ptr = dpct::get_raw_pointer(d[0].data());
