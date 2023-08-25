// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0

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

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::set_intersection | FileCheck %s -check-prefix=set_intersection
// set_intersection:  oneapi::dpl::set_intersection(oneapi::dpl::execution::seq, A1, A1 + 6, A2, A2 + 7, result);
// set_intersection-NEXT:  oneapi::dpl::set_intersection(oneapi::dpl::execution::seq, A1, A1 + 6, A2, A2 + 7, result);
// set_intersection-NEXT:  oneapi::dpl::set_intersection(oneapi::dpl::execution::seq, A1, A1 + 6, A2, A2 + 7, result, std::greater<int>());
// set_intersection-NEXT:  oneapi::dpl::set_intersection(oneapi::dpl::execution::seq, A1, A1 + 6, A2, A2 + 7, result, std::greater<int>());
// set_intersection-NEXT:  oneapi::dpl::set_intersection(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin());
// set_intersection-NEXT:  oneapi::dpl::set_intersection(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin());
// set_intersection-NEXT:  oneapi::dpl::set_intersection(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin(), std::greater<int>());
// set_intersection-NEXT:  oneapi::dpl::set_intersection(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin(), std::greater<int>());
// set_intersection-NEXT:  oneapi::dpl::set_intersection(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin());
// set_intersection-NEXT:  oneapi::dpl::set_intersection(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin());
// set_intersection-NEXT:  oneapi::dpl::set_intersection(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin(), std::greater<int>());
// set_intersection-NEXT:  oneapi::dpl::set_intersection(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin(), std::greater<int>());
// set_intersection-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::is_sorted_until | FileCheck %s -check-prefix=is_sorted_until
// is_sorted_until:  oneapi::dpl::is_sorted_until(oneapi::dpl::execution::seq, A, A + 8);
// is_sorted_until-NEXT:  oneapi::dpl::is_sorted_until(oneapi::dpl::execution::seq, A, A + 8);
// is_sorted_until-NEXT:  oneapi::dpl::is_sorted_until(oneapi::dpl::execution::seq, A, A + 8, comp);
// is_sorted_until-NEXT:  oneapi::dpl::is_sorted_until(oneapi::dpl::execution::seq, A, A + 8, comp);
// is_sorted_until-NEXT:  oneapi::dpl::is_sorted_until(oneapi::dpl::execution::seq, h_A.begin(), h_A.end());
// is_sorted_until-NEXT:  oneapi::dpl::is_sorted_until(oneapi::dpl::execution::seq, h_A.begin(), h_A.end());
// is_sorted_until-NEXT:  oneapi::dpl::is_sorted_until(oneapi::dpl::execution::seq, h_A.begin(), h_A.end(), comp);
// is_sorted_until-NEXT:  oneapi::dpl::is_sorted_until(oneapi::dpl::execution::seq, h_A.begin(), h_A.end(), comp);
// is_sorted_until-NEXT:  oneapi::dpl::is_sorted_until(oneapi::dpl::execution::make_device_policy(q_ct1), d_A.begin(), d_A.end());
// is_sorted_until-NEXT:  oneapi::dpl::is_sorted_until(oneapi::dpl::execution::make_device_policy(q_ct1), d_A.begin(), d_A.end());
// is_sorted_until-NEXT:  oneapi::dpl::is_sorted_until(oneapi::dpl::execution::make_device_policy(q_ct1), d_A.begin(), d_A.end(), comp);
// is_sorted_until-NEXT:  oneapi::dpl::is_sorted_until(oneapi::dpl::execution::make_device_policy(q_ct1), d_A.begin(), d_A.end(), comp);
// is_sorted_until-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::is_partitioned | FileCheck %s -check-prefix=is_partitioned
// is_partitioned:  oneapi::dpl::is_partitioned(oneapi::dpl::execution::seq, A, A + 10, is_even());
// is_partitioned-NEXT:  oneapi::dpl::is_partitioned(oneapi::dpl::execution::seq, A, A + 10, is_even());
// is_partitioned-NEXT:  oneapi::dpl::is_partitioned(oneapi::dpl::execution::seq, h_A.begin(), h_A.end(), is_even());
// is_partitioned-NEXT:  oneapi::dpl::is_partitioned(oneapi::dpl::execution::seq, h_A.begin(), h_A.end(), is_even());
// is_partitioned-NEXT:  oneapi::dpl::is_partitioned(oneapi::dpl::execution::make_device_policy(q_ct1), d_A.begin(), d_A.end(), is_even());
// is_partitioned-NEXT:  oneapi::dpl::is_partitioned(oneapi::dpl::execution::make_device_policy(q_ct1), d_A.begin(), d_A.end(), is_even());
// is_partitioned-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::none_of | FileCheck %s -check-prefix=none_of
// none_of:   oneapi::dpl::none_of(oneapi::dpl::execution::seq, A, A + 2, oneapi::dpl::identity());
// none_of-NEXT:  oneapi::dpl::none_of(oneapi::dpl::execution::seq, A, A + 2, oneapi::dpl::identity());
// none_of-NEXT:   oneapi::dpl::none_of(oneapi::dpl::execution::seq, h_A.begin(), h_A.begin() + 2, oneapi::dpl::identity());
// none_of-NEXT:   oneapi::dpl::none_of(oneapi::dpl::execution::seq, h_A.begin(), h_A.begin() + 2, oneapi::dpl::identity());
// none_of-NEXT:   oneapi::dpl::none_of(oneapi::dpl::execution::make_device_policy(q_ct1), d_A.begin(), d_A.begin() + 2, oneapi::dpl::identity());
// none_of-NEXT:   oneapi::dpl::none_of(oneapi::dpl::execution::make_device_policy(q_ct1), d_A.begin(), d_A.begin() + 2, oneapi::dpl::identity());
// none_of-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::all_of | FileCheck %s -check-prefix=all_of
// all_of:  oneapi::dpl::all_of(oneapi::dpl::execution::seq, A, A + 2, oneapi::dpl::identity());
// all_of-NEXT:  oneapi::dpl::all_of(oneapi::dpl::execution::seq, A, A + 2, oneapi::dpl::identity());
// all_of-NEXT:  oneapi::dpl::all_of(oneapi::dpl::execution::seq, h_A.begin(), h_A.begin() + 2, oneapi::dpl::identity());
// all_of-NEXT:  oneapi::dpl::all_of(oneapi::dpl::execution::seq, h_A.begin(), h_A.begin() + 2, oneapi::dpl::identity());
// all_of-NEXT:  oneapi::dpl::all_of(oneapi::dpl::execution::make_device_policy(q_ct1), d_A.begin(), d_A.begin() + 2, oneapi::dpl::identity());
// all_of-NEXT:  oneapi::dpl::all_of(oneapi::dpl::execution::make_device_policy(q_ct1), d_A.begin(), d_A.begin() + 2, oneapi::dpl::identity());
// all_of-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::uninitialized_fill_n | FileCheck %s -check-prefix=uninitialized_fill_n
// uninitialized_fill_n:  oneapi::dpl::uninitialized_fill_n(oneapi::dpl::execution::make_device_policy(q_ct1), d_array, N, int_val);
// uninitialized_fill_n-NEXT:  oneapi::dpl::uninitialized_fill_n(oneapi::dpl::execution::make_device_policy(q_ct1), d_array, N, int_val);
// uninitialized_fill_n-NEXT:  oneapi::dpl::uninitialized_fill_n(oneapi::dpl::execution::seq, data, N, val);
// uninitialized_fill_n-NEXT:  oneapi::dpl::uninitialized_fill_n(oneapi::dpl::execution::seq, data, N, val);
// uninitialized_fill_n-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::swap_ranges | FileCheck %s -check-prefix=swap_ranges
// swap_ranges:  oneapi::dpl::swap_ranges(oneapi::dpl::execution::make_device_policy(q_ct1), d_v1.begin(), d_v1.end(), d_v2.begin());
// swap_ranges-NEXT:  oneapi::dpl::swap_ranges(oneapi::dpl::execution::make_device_policy(q_ct1), d_v1.begin(), d_v1.end(), d_v2.begin());
// swap_ranges-NEXT:  oneapi::dpl::swap_ranges(oneapi::dpl::execution::seq, h_v1.begin(), h_v1.end(), h_v2.begin());
// swap_ranges-NEXT:  oneapi::dpl::swap_ranges(oneapi::dpl::execution::seq, h_v1.begin(), h_v1.end(), h_v2.begin());
// swap_ranges-NEXT:  oneapi::dpl::swap_ranges(oneapi::dpl::execution::seq, v1, v1 + 2, v2);
// swap_ranges-NEXT:  oneapi::dpl::swap_ranges(oneapi::dpl::execution::seq, v1, v1 + 2, v2);
// swap_ranges-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::set_symmetric_difference_by_key | FileCheck %s -check-prefix=set_symmetric_difference_by_key
// set_symmetric_difference_by_key:  dpct::set_symmetric_difference(oneapi::dpl::execution::seq, A_keys, A_keys + 7, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result);
// set_symmetric_difference_by_key-NEXT:  dpct::set_symmetric_difference(oneapi::dpl::execution::seq, A_keys, A_keys + 7, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result);
// set_symmetric_difference_by_key-NEXT:  dpct::set_symmetric_difference(oneapi::dpl::execution::seq, A_keys, A_keys + 7, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result, Compare());
// set_symmetric_difference_by_key-NEXT:  dpct::set_symmetric_difference(oneapi::dpl::execution::seq, A_keys, A_keys + 7, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result, Compare());
// set_symmetric_difference_by_key-NEXT:  dpct::set_symmetric_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_A_keys.begin(), d_A_keys.end(), d_B_keys.begin(), d_B_keys.end(), d_A_vals.begin(), d_B_vals.begin(), d_keys_result.begin(), d_vals_result.begin());
// set_symmetric_difference_by_key-NEXT:  dpct::set_symmetric_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_A_keys.begin(), d_A_keys.end(), d_B_keys.begin(), d_B_keys.end(), d_A_vals.begin(), d_B_vals.begin(), d_keys_result.begin(), d_vals_result.begin());
// set_symmetric_difference_by_key-NEXT:  dpct::set_symmetric_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_A_keys.begin(), d_A_keys.end(), d_B_keys.begin(), d_B_keys.end(), d_A_vals.begin(), d_B_vals.begin(), d_keys_result.begin(), d_vals_result.begin(), Compare());
// set_symmetric_difference_by_key-NEXT:  dpct::set_symmetric_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_A_keys.begin(), d_A_keys.end(), d_B_keys.begin(), d_B_keys.end(), d_A_vals.begin(), d_B_vals.begin(), d_keys_result.begin(), d_vals_result.begin(), Compare());
// set_symmetric_difference_by_key-NEXT:  dpct::set_symmetric_difference(oneapi::dpl::execution::seq, h_A_keys.begin(), h_A_keys.end(), h_B_keys.begin(), h_B_keys.end(), h_A_vals.begin(), h_B_vals.begin(), h_keys_result.begin(), h_vals_result.begin());
// set_symmetric_difference_by_key-NEXT:  dpct::set_symmetric_difference(oneapi::dpl::execution::seq, h_A_keys.begin(), h_A_keys.end(), h_B_keys.begin(), h_B_keys.end(), h_A_vals.begin(), h_B_vals.begin(), h_keys_result.begin(), h_vals_result.begin());
// set_symmetric_difference_by_key-NEXT:  dpct::set_symmetric_difference(oneapi::dpl::execution::seq, h_A_keys.begin(), h_A_keys.end(), h_B_keys.begin(), h_B_keys.end(), h_A_vals.begin(), h_B_vals.begin(), h_keys_result.begin(), h_vals_result.begin(), Compare());
// set_symmetric_difference_by_key-NEXT:  dpct::set_symmetric_difference(oneapi::dpl::execution::seq, h_A_keys.begin(), h_A_keys.end(), h_B_keys.begin(), h_B_keys.end(), h_A_vals.begin(), h_B_vals.begin(), h_keys_result.begin(), h_vals_result.begin(), Compare());
// set_symmetric_difference_by_key-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::set_symmetric_difference | FileCheck %s -check-prefix=set_symmetric_difference
// set_symmetric_difference:  oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::seq, A1, A1 + 4, A2, A2 + 2, result);
// set_symmetric_difference-NEXT:  oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::seq, A1, A1 + 4, A2, A2 + 2, result);
// set_symmetric_difference-NEXT:  oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::seq, A1, A1 + 5, A2, A2 + 5, result, Compare());
// set_symmetric_difference-NEXT:  oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::seq, A1, A1 + 5, A2, A2 + 5, result, Compare());
// set_symmetric_difference-NEXT:  oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin());
// set_symmetric_difference-NEXT:  oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin());
// set_symmetric_difference-NEXT:  oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin(), oneapi::dpl::less<int>());
// set_symmetric_difference-NEXT:  oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin(), oneapi::dpl::less<int>());
// set_symmetric_difference-NEXT:  oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin());
// set_symmetric_difference-NEXT:  oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin());
// set_symmetric_difference-NEXT:  oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin(), oneapi::dpl::less<int>());
// set_symmetric_difference-NEXT:  oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin(), oneapi::dpl::less<int>());
// set_symmetric_difference-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::equal | FileCheck %s -check-prefix=equal
// equal:  oneapi::dpl::equal(oneapi::dpl::execution::seq, A1, A1 + N, A2);
// equal-NEXT:  oneapi::dpl::equal(oneapi::dpl::execution::seq, A1, A1 + N, A2);
// equal-NEXT:  oneapi::dpl::equal(oneapi::dpl::execution::seq, x, x + N, y, compare_modulo_two());
// equal-NEXT:  oneapi::dpl::equal(oneapi::dpl::execution::seq, x, x + N, y, compare_modulo_two());
// equal-NEXT:  oneapi::dpl::equal(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin());
// equal-NEXT:  oneapi::dpl::equal(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin());
// equal-NEXT:  oneapi::dpl::equal(oneapi::dpl::execution::seq, h_x.begin(), h_x.end(), h_y.begin(), compare_modulo_two());
// equal-NEXT:  oneapi::dpl::equal(oneapi::dpl::execution::seq, h_x.begin(), h_x.end(), h_y.begin(), compare_modulo_two());
// equal-NEXT:  oneapi::dpl::equal(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin());
// equal-NEXT:  oneapi::dpl::equal(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin());
// equal-NEXT:  oneapi::dpl::equal(oneapi::dpl::execution::make_device_policy(q_ct1), d_x.begin(), d_x.end(), d_y.begin(), compare_modulo_two());
// equal-NEXT:  oneapi::dpl::equal(oneapi::dpl::execution::make_device_policy(q_ct1), d_x.begin(), d_x.end(), d_y.begin(), compare_modulo_two());
// equal-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::uninitialized_copy_n | FileCheck %s -check-prefix=uninitialized_copy_n
// uninitialized_copy_n:  oneapi::dpl::uninitialized_copy_n(oneapi::dpl::execution::make_device_policy(q_ct1), d_input.begin(), N, d_array);
// uninitialized_copy_n-NEXT:  oneapi::dpl::uninitialized_copy_n(oneapi::dpl::execution::seq, h_data, N, h_array);
// uninitialized_copy_n-NEXT:  oneapi::dpl::uninitialized_copy_n(oneapi::dpl::execution::make_device_policy(q_ct1), d_input.begin(), N, d_array);
// uninitialized_copy_n-NEXT:  oneapi::dpl::uninitialized_copy_n(oneapi::dpl::execution::seq, h_data, N, h_array);
// uninitialized_copy_n-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::uninitialized_copy | FileCheck %s -check-prefix=uninitialized_copy
// uninitialized_copy:  oneapi::dpl::uninitialized_copy(oneapi::dpl::execution::make_device_policy(q_ct1), d_input.begin(), d_input.end(), d_array);
// uninitialized_copy-NEXT:  oneapi::dpl::uninitialized_copy(oneapi::dpl::execution::seq, data, data + N, array);
// uninitialized_copy-NEXT:  oneapi::dpl::uninitialized_copy(oneapi::dpl::execution::make_device_policy(q_ct1), d_input.begin(), d_input.end(), d_array);
// uninitialized_copy-NEXT:  oneapi::dpl::uninitialized_copy(oneapi::dpl::execution::seq, data, data + N, h_array);
// uninitialized_copy-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::transform_inclusive_scan | FileCheck %s -check-prefix=transform_inclusive_scan
// transform_inclusive_scan:  oneapi::dpl::transform_inclusive_scan(oneapi::dpl::execution::seq, data, data + N, data, binary_op, unary_op);
// transform_inclusive_scan-NEXT:  oneapi::dpl::transform_inclusive_scan(oneapi::dpl::execution::seq, h_vec_data.begin(), h_vec_data.end(), h_vec_data.begin(), binary_op, unary_op);
// transform_inclusive_scan-NEXT:  oneapi::dpl::transform_inclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), d_vec_data.begin(), d_vec_data.end(), d_vec_data.begin(), binary_op, unary_op);
// transform_inclusive_scan-NEXT:  oneapi::dpl::transform_inclusive_scan(oneapi::dpl::execution::seq, data, data + N, data, binary_op, unary_op);
// transform_inclusive_scan-NEXT:  oneapi::dpl::transform_inclusive_scan(oneapi::dpl::execution::seq, h_vec_data.begin(), h_vec_data.end(), h_vec_data.begin(), binary_op, unary_op);
// transform_inclusive_scan-NEXT:  oneapi::dpl::transform_inclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), d_vec_data.begin(), d_vec_data.end(), d_vec_data.begin(), binary_op, unary_op);
// transform_inclusive_scan-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::equal_range | FileCheck %s -check-prefix=equal_range
// equal_range:  dpct::equal_range(oneapi::dpl::execution::make_device_policy(q_ct1), device_vec.begin(), device_vec.end(), 0);
// equal_range-NEXT:  dpct::equal_range(oneapi::dpl::execution::make_device_policy(q_ct1), device_vec.begin(), device_vec.end(), 0);
// equal_range-NEXT:  dpct::equal_range(oneapi::dpl::execution::make_device_policy(q_ct1), device_vec.begin(), device_vec.end(), 0, oneapi::dpl::less<int>());
// equal_range-NEXT:  dpct::equal_range(oneapi::dpl::execution::make_device_policy(q_ct1), device_vec.begin(), device_vec.end(), 0, oneapi::dpl::less<int>());
// equal_range-NEXT:  dpct::equal_range(oneapi::dpl::execution::seq, host_vec.begin(), host_vec.end(), 0);
// equal_range-NEXT:  dpct::equal_range(oneapi::dpl::execution::seq, host_vec.begin(), host_vec.end(), 0);
// equal_range-NEXT:  dpct::equal_range(oneapi::dpl::execution::seq, host_vec.begin(), host_vec.end(), 0, oneapi::dpl::less<int>());
// equal_range-NEXT:  dpct::equal_range(oneapi::dpl::execution::seq, host_vec.begin(), host_vec.end(), 0, oneapi::dpl::less<int>());
// equal_range-NEXT:  dpct::equal_range(oneapi::dpl::execution::seq, data, data + N, 0);
// equal_range-NEXT:  dpct::equal_range(oneapi::dpl::execution::seq, data, data + N, 0);
// equal_range-NEXT:  dpct::equal_range(oneapi::dpl::execution::seq, data, data + N, 0, oneapi::dpl::less<int>());
// equal_range-NEXT:  dpct::equal_range(oneapi::dpl::execution::seq, data, data + N, 0, oneapi::dpl::less<int>());
// equal_range-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::reverse | FileCheck %s -check-prefix=reverse
// reverse:   oneapi::dpl::reverse(oneapi::dpl::execution::make_device_policy(q_ct1), device_data.begin(), device_data.end());
// reverse-NEXT:  oneapi::dpl::reverse(oneapi::dpl::execution::seq, host_data.begin(), host_data.end());
// reverse-NEXT:  oneapi::dpl::reverse(oneapi::dpl::execution::seq, data, data + N);
// reverse-NEXT:  oneapi::dpl::reverse(oneapi::dpl::execution::make_device_policy(q_ct1), device_data.begin(), device_data.end());
// reverse-NEXT:  oneapi::dpl::reverse(oneapi::dpl::execution::seq, host_data.begin(), host_data.end());
// reverse-NEXT:  oneapi::dpl::reverse(oneapi::dpl::execution::seq, data, data + N);
