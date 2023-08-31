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

  int A1[7] = {0, 2, 4, 6, 8, 10, 12};
  int A2[5] = {1, 3, 5, 7, 9};
  int result[12];
  int *result_end;
  thrust::device_vector<int> d_A1(A1, A1 + 7);
  thrust::device_vector<int> d_A2(A2, A2 + 5);
  thrust::device_vector<int> d_result(11);
  thrust::device_vector<int>::iterator d_result_iter_end;
  thrust::host_vector<int> h_A1(A1, A1 + 7);
  thrust::host_vector<int> h_A2(A2, A2 + 5);
  thrust::host_vector<int> h_result(12);
  thrust::host_vector<int>::iterator h_result_iter_end;

  // Start
  /*1*/ thrust::set_union(thrust::host /*const thrust::detail::execution_policy_base<
                                    DerivedPolicy > &*/,
                    A1 /*InputIterator1*/, A1 + 7 /*InputIterator1*/,
                    A2 /*InputIterator2 */, A2 + 5 /*InputIterator2 */,
                    result /*OutputIterator */);
  /*2*/ thrust::set_union(A1 /*InputIterator1*/, A1 + 7 /*InputIterator1*/,
                    A2 /*InputIterator2 */, A2 + 5 /*InputIterator2 */,
                    result /*OutputIterator */);
  /*3*/ thrust::set_union(thrust::host /*const thrust::detail::execution_policy_base<
                                    DerivedPolicy > &*/,
                    A1 /*InputIterator1*/, A1 + 7 /*InputIterator1*/,
                    A2 /*InputIterator2 */, A2 + 5 /*InputIterator2 */,
                    result /*OutputIterator */, thrust::greater<int>());
  /*4*/ thrust::set_union(A1 /*InputIterator1*/, A1 + 7 /*InputIterator1*/,
                    A2 /*InputIterator2 */, A2 + 5 /*InputIterator2 */,
                    result /*OutputIterator */,
                    thrust::greater<int>() /*StrictWeakCompare */);
  /*5*/ thrust::set_union(
      thrust::device /*const thrust::detail::execution_policy_base<
                        DerivedPolicy > &*/,
      d_A1.begin() /*InputIterator1*/, d_A1.end() /*InputIterator1*/,
      d_A2.begin() /*InputIterator2 */, d_A2.end() /*InputIterator2 */,
      d_result.begin() /*OutputIterator */);
  /*6*/ thrust::set_union(
      d_A1.begin() /*InputIterator1*/, d_A1.end() /*InputIterator1*/,
      d_A2.begin() /*InputIterator2 */, d_A2.end() /*InputIterator2 */,
      d_result.begin() /*OutputIterator */);
  /*7*/ thrust::set_union(
      thrust::device /*const thrust::detail::execution_policy_base<
                        DerivedPolicy > &*/,
      d_A1.begin() /*InputIterator1*/, d_A1.end() /*InputIterator1*/,
      d_A2.begin() /*InputIterator2 */, d_A2.end() /*InputIterator2 */,
      d_result.begin() /*OutputIterator */, thrust::greater<int>());
  /*8*/ thrust::set_union(
      d_A1.begin() /*InputIterator1*/, d_A1.end() /*InputIterator1*/,
      d_A2.begin() /*InputIterator2 */, d_A2.end() /*InputIterator2 */,
      d_result.begin() /*OutputIterator */,
      thrust::greater<int>() /*StrictWeakCompare */);
  /*9*/ thrust::set_union(
      thrust::host /*const thrust::detail::execution_policy_base<
                      DerivedPolicy > &*/,
      h_A1.begin() /*InputIterator1*/, h_A1.end() /*InputIterator1*/,
      h_A2.begin() /*InputIterator2 */, h_A2.end() /*InputIterator2 */,
      h_result.begin() /*OutputIterator */);
  /*10*/ thrust::set_union(
      h_A1.begin() /*InputIterator1*/, h_A1.end() /*InputIterator1*/,
      h_A2.begin() /*InputIterator2 */, h_A2.end() /*InputIterator2 */,
      h_result.begin() /*OutputIterator */);
  /*11*/ thrust::set_union(
      thrust::host /*const thrust::detail::execution_policy_base<
                      DerivedPolicy > &*/,
      h_A1.begin() /*InputIterator1*/, h_A1.end() /*InputIterator1*/,
      h_A2.begin() /*InputIterator2 */, h_A2.end() /*InputIterator2 */,
      h_result.begin() /*OutputIterator */,
      thrust::greater<int>() /*StrictWeakCompare */);
  /*12*/ thrust::set_union(
      h_A1.begin() /*InputIterator1*/, h_A1.end() /*InputIterator1*/,
      h_A2.begin() /*InputIterator2 */, h_A2.end() /*InputIterator2 */,
      h_result.begin() /*OutputIterator */,
      thrust::greater<int>() /*StrictWeakCompare */);
  // End
}
