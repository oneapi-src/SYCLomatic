#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
#include <thrust/partition.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>

void test() {

  int A1[6] = {1, 3, 5, 7, 9, 11};
  int A2[7] = {1, 1, 2, 3, 5, 8, 13};
  int result[3];
  int *result_end;
  thrust::device_vector<int> d_A1(A1, A1 + 6);
  thrust::device_vector<int> d_A2(A2, A2 + 7);
  thrust::device_vector<int> d_result(3);
  thrust::device_vector<int>::iterator d_end;
  thrust::host_vector<int> h_A1(A1, A1 + 6);
  thrust::host_vector<int> h_A2(A2, A2 + 7);
  thrust::host_vector<int> h_result(3);
  thrust::host_vector<int>::iterator h_end;

  // Start
  thrust::set_intersection(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > &*/,
      A1 /*InputIterator1 */, A1 + 6 /*InputIterator1 */,
      A2 /*InputIterator2 */, A2 + 7 /*InputIterator2 */, result);
  thrust::set_intersection(A1 /*InputIterator1 */, A1 + 6 /*InputIterator1 */,
                           A2 /*InputIterator2 */, A2 + 7 /*InputIterator2 */,
                           result /*OutputIterator */);
  thrust::set_intersection(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > &*/,
      A1 /*InputIterator1 */, A1 + 6 /*InputIterator1 */,
      A2 /*InputIterator2 */, A2 + 7 /*InputIterator2 */,
      result /*OutputIterator */,
      thrust::greater<int>() /*StrictWeakCompare */);
  thrust::set_intersection(A1 /*InputIterator1 */, A1 + 6 /*InputIterator1 */,
                           A2 /*InputIterator2 */, A2 + 7 /*InputIterator2 */,
                           result /*OutputIterator */,
                           thrust::greater<int>() /*StrictWeakCompare */);
  thrust::set_intersection(
      thrust::device /*const thrust::detail::execution_policy_base<
                        DerivedPolicy > &*/,
      d_A1.begin() /*InputIterator1 */, d_A1.end() /*InputIterator1 */,
      d_A2.begin() /*InputIterator2 */, d_A2.end() /*InputIterator2 */,
      d_result.begin() /*OutputIterator */);
  thrust::set_intersection(
      d_A1.begin() /*InputIterator1 */, d_A1.end() /*InputIterator1 */,
      d_A2.begin() /*InputIterator2 */, d_A2.end() /*InputIterator2 */,
      d_result.begin() /*OutputIterator */);
  thrust::set_intersection(
      thrust::device /*const thrust::detail::execution_policy_base<
                        DerivedPolicy > &*/,
      d_A1.begin() /*InputIterator1 */, d_A1.end() /*InputIterator1 */,
      d_A2.begin() /*InputIterator2 */, d_A2.end() /*InputIterator2 */,
      d_result.begin() /*OutputIterator */, thrust::greater<int>());
  thrust::set_intersection(
      d_A1.begin() /*InputIterator1 */, d_A1.end() /*InputIterator1 */,
      d_A2.begin() /*InputIterator2 */, d_A2.end() /*InputIterator2 */,
      d_result.begin() /*OutputIterator */,
      thrust::greater<int>() /*StrictWeakCompare */);
  thrust::set_intersection(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > &*/,
      h_A1.begin() /*InputIterator1 */, h_A1.end() /*InputIterator1 */,
      h_A2.begin() /*InputIterator2 */, h_A2.end() /*InputIterator2 */,
      h_result.begin() /*OutputIterator */);
  thrust::set_intersection(
      h_A1.begin() /*InputIterator1 */, h_A1.end() /*InputIterator1 */,
      h_A2.begin() /*InputIterator2 */, h_A2.end() /*InputIterator2 */,
      h_result.begin() /*OutputIterator */);
  thrust::set_intersection(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > &*/,
      h_A1.begin() /*InputIterator1 */, h_A1.end() /*InputIterator1 */,
      h_A2.begin() /*InputIterator2 */, h_A2.end() /*InputIterator2 */,
      h_result.begin() /*OutputIterator */,
      thrust::greater<int>() /*StrictWeakCompare */);
  thrust::set_intersection(
      h_A1.begin() /*InputIterator1 */, h_A1.end() /*InputIterator1 */,
      h_A2.begin() /*InputIterator2 */, h_A2.end() /*InputIterator2 */,
      h_result.begin() /*OutputIterator */,
      thrust::greater<int>() /*StrictWeakCompare */);
  // End
}
