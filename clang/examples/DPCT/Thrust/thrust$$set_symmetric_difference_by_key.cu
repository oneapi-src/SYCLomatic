#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/set_operations.h>

struct Compare {
  __host__ __device__ bool operator()(const int &a, const int &b) {
    return a < b;
  }
};

void test() {

  int A_keys[7] = {0, 1, 2, 2, 4, 6, 7};
  int A_vals[7] = {0, 0, 0, 0, 0, 0, 0};
  int B_keys[5] = {1, 1, 2, 5, 8};
  int B_vals[5] = {1, 1, 1, 1, 1};
  int keys_result[8];
  int vals_result[8];
  thrust::device_vector<int> d_keys_result(8);
  thrust::device_vector<int> d_vals_result(8);
  thrust::device_vector<int> d_A_keys(A_keys, A_keys + 7);
  thrust::device_vector<int> d_A_vals(A_vals, A_vals + 7);
  thrust::device_vector<int> d_B_keys(B_keys, B_keys + 5);
  thrust::device_vector<int> d_B_vals(B_vals, B_vals + 5);
  thrust::host_vector<int> h_keys_result(8);
  thrust::host_vector<int> h_vals_result(8);
  thrust::host_vector<int> h_A_keys(A_keys, A_keys + 7);
  thrust::host_vector<int> h_A_vals(A_vals, A_vals + 7);
  thrust::host_vector<int> h_B_keys(B_keys, B_keys + 5);
  thrust::host_vector<int> h_B_vals(B_vals, B_vals + 5);

  // Start
  thrust::set_symmetric_difference_by_key(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > &*/,
      A_keys /*InputIterator1 */, A_keys + 7 /*InputIterator1 */,
      B_keys /*InputIterator2*/, B_keys + 5 /*InputIterator2*/,
      A_vals /*InputIterator3*/, B_vals /*InputIterator4*/,
      keys_result /*OutputIterator1*/, vals_result);
  thrust::set_symmetric_difference_by_key(
      A_keys /*InputIterator1 */, A_keys + 7 /*InputIterator1 */,
      B_keys /*InputIterator2*/, B_keys + 5 /*InputIterator2*/,
      A_vals /*InputIterator3*/, B_vals, keys_result /*OutputIterator1*/,
      vals_result /*OutputIterator2*/);
  thrust::set_symmetric_difference_by_key(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > &*/,
      A_keys /*InputIterator1 */, A_keys + 7 /*InputIterator1 */,
      B_keys /*InputIterator2*/, B_keys + 5 /*InputIterator2*/,
      A_vals /*InputIterator3*/, B_vals /*InputIterator4*/,
      keys_result /*OutputIterator1*/, vals_result /*OutputIterator2*/,
      Compare() /*StrictWeakCompare*/);
  thrust::set_symmetric_difference_by_key(
      A_keys /*InputIterator1 */, A_keys + 7 /*InputIterator1 */,
      B_keys /*InputIterator2*/, B_keys + 5 /*InputIterator2*/,
      A_vals /*InputIterator3*/, B_vals /*InputIterator4*/,
      keys_result /*OutputIterator1*/, vals_result /*OutputIterator2*/,
      Compare() /*StrictWeakCompare*/);
  thrust::set_symmetric_difference_by_key(
      thrust::device /*const thrust::detail::execution_policy_base<
                        DerivedPolicy > &*/,
      d_A_keys.begin() /*InputIterator1*/, d_A_keys.end() /*InputIterator1 */,
      d_B_keys.begin() /*InputIterator2*/, d_B_keys.end() /*InputIterator2*/,
      d_A_vals.begin() /*InputIterator3*/, d_B_vals.begin() /*InputIterator4*/,
      d_keys_result.begin() /*OutputIterator1*/,
      d_vals_result.begin() /*OutputIterator2*/);
  thrust::set_symmetric_difference_by_key(
      d_A_keys.begin() /*InputIterator1 */, d_A_keys.end() /*InputIterator1 */,
      d_B_keys.begin() /*InputIterator2*/, d_B_keys.end() /*InputIterator2*/,
      d_A_vals.begin() /*InputIterator3*/, d_B_vals.begin() /*InputIterator4*/,
      d_keys_result.begin() /*OutputIterator1*/,
      d_vals_result.begin() /*OutputIterator2*/);
  thrust::set_symmetric_difference_by_key(
      thrust::device /*const thrust::detail::execution_policy_base<
                        DerivedPolicy > &*/,
      d_A_keys.begin() /*InputIterator1 */, d_A_keys.end() /*InputIterator1 */,
      d_B_keys.begin() /*InputIterator2*/, d_B_keys.end() /*InputIterator2*/,
      d_A_vals.begin() /*InputIterator3*/, d_B_vals.begin() /*InputIterator4*/,
      d_keys_result.begin() /*OutputIterator1*/,
      d_vals_result.begin() /*OutputIterator2*/,
      Compare() /*StrictWeakCompare*/);
  thrust::set_symmetric_difference_by_key(
      d_A_keys.begin() /*InputIterator1 */, d_A_keys.end() /*InputIterator1 */,
      d_B_keys.begin() /*InputIterator2*/, d_B_keys.end() /*InputIterator2*/,
      d_A_vals.begin() /*InputIterator3*/, d_B_vals.begin() /*InputIterator4*/,
      d_keys_result.begin() /*OutputIterator1*/,
      d_vals_result.begin() /*OutputIterator2*/,
      Compare() /*StrictWeakCompare*/);
  thrust::set_symmetric_difference_by_key(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > &*/,
      h_A_keys.begin() /*InputIterator1 */, h_A_keys.end() /*InputIterator1 */,
      h_B_keys.begin() /*InputIterator2*/, h_B_keys.end() /*InputIterator2*/,
      h_A_vals.begin() /*InputIterator3*/, h_B_vals.begin() /*InputIterator4*/,
      h_keys_result.begin() /*OutputIterator1*/,
      h_vals_result.begin() /*OutputIterator2*/);
  thrust::set_symmetric_difference_by_key(
      h_A_keys.begin() /*InputIterator1 */, h_A_keys.end() /*InputIterator1 */,
      h_B_keys.begin() /*InputIterator2*/, h_B_keys.end() /*InputIterator2*/,
      h_A_vals.begin() /*InputIterator3*/, h_B_vals.begin() /*InputIterator4*/,
      h_keys_result.begin() /*OutputIterator1*/,
      h_vals_result.begin() /*OutputIterator2*/);
  thrust::set_symmetric_difference_by_key(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > &*/,
      h_A_keys.begin() /*InputIterator1 */, h_A_keys.end() /*InputIterator1 */,
      h_B_keys.begin() /*InputIterator2*/, h_B_keys.end() /*InputIterator2*/,
      h_A_vals.begin() /*InputIterator3*/, h_B_vals.begin() /*InputIterator4*/,
      h_keys_result.begin() /*OutputIterator1*/,
      h_vals_result.begin() /*OutputIterator2*/, Compare());
  thrust::set_symmetric_difference_by_key(
      h_A_keys.begin() /*InputIterator1 */, h_A_keys.end() /*InputIterator1 */,
      h_B_keys.begin() /*InputIterator2*/, h_B_keys.end() /*InputIterator2*/,
      h_A_vals.begin() /*InputIterator3*/, h_B_vals.begin() /*InputIterator4*/,
      h_keys_result.begin() /*OutputIterator1*/,
      h_vals_result.begin() /*OutputIterator2*/,
      Compare() /*StrictWeakCompare*/);
  // End
}
