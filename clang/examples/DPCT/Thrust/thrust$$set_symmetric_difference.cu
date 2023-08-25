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

  int A1[7] = {0, 1, 2, 2, 4, 6, 7};
  int A2[5] = {1, 1, 2, 5, 8};
  int result[8];
  thrust::device_vector<int> d_A1(A1, A1 + 7);
  thrust::device_vector<int> d_A2(A2, A2 + 5);
  thrust::device_vector<int> d_result(8);
  thrust::host_vector<int> h_A1(A1, A1 + 7);
  thrust::host_vector<int> h_A2(A2, A2 + 5);
  thrust::host_vector<int> h_result(8);

  // Start
  thrust::set_symmetric_difference(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > &*/,
      A1 /*InputIterator1*/, A1 + 4 /*InputIterator1*/, A2 /*InputIterator2*/,
      A2 + 2 /*InputIterator2*/, result /*OutputIterator*/);
  thrust::set_symmetric_difference(
      A1 /*InputIterator1*/, A1 + 4 /*InputIterator1*/, A2 /*InputIterator2*/,
      A2 + 2 /*InputIterator2*/, result /*OutputIterator*/);
  thrust::set_symmetric_difference(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > &*/,
      A1 /*InputIterator1*/, A1 + 5 /*InputIterator1*/, A2 /*InputIterator2*/,
      A2 + 5 /*InputIterator2*/, result /*OutputIterator*/,
      Compare() /*StrictWeakCompare*/);
  thrust::set_symmetric_difference(
      A1 /*InputIterator1*/, A1 + 5 /*InputIterator1*/, A2 /*InputIterator2*/,
      A2 + 5 /*InputIterator2*/, result /*OutputIterator*/,
      Compare() /*StrictWeakCompare*/);
  thrust::set_symmetric_difference(
      thrust::device /*const thrust::detail::execution_policy_base<
                        DerivedPolicy > &*/,
      d_A1.begin() /*InputIterator1*/, d_A1.end() /*InputIterator1*/,
      d_A2.begin() /*InputIterator2*/, d_A2.end() /*InputIterator2*/,
      d_result.begin());
  thrust::set_symmetric_difference(
      d_A1.begin() /*InputIterator1*/, d_A1.end() /*InputIterator1*/,
      d_A2.begin() /*InputIterator2*/, d_A2.end() /*InputIterator2*/,
      d_result.begin());
  thrust::set_symmetric_difference(
      thrust::device /*const thrust::detail::execution_policy_base<
                        DerivedPolicy > &*/,
      d_A1.begin() /*InputIterator1*/, d_A1.end() /*InputIterator1*/,
      d_A2.begin() /*InputIterator2*/, d_A2.end() /*InputIterator2*/,
      d_result.begin() /*OutputIterator*/, thrust::less<int>());
  thrust::set_symmetric_difference(
      d_A1.begin() /*InputIterator1*/, d_A1.end() /*InputIterator1*/,
      d_A2.begin() /*InputIterator2*/, d_A2.end() /*InputIterator2*/,
      d_result.begin() /*OutputIterator*/, thrust::less<int>());
  thrust::set_symmetric_difference(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > &*/,
      h_A1.begin() /*InputIterator1*/, h_A1.end() /*InputIterator1*/,
      h_A2.begin() /*InputIterator2*/, h_A2.end() /*InputIterator2*/,
      h_result.begin());
  thrust::set_symmetric_difference(
      h_A1.begin() /*InputIterator1*/, h_A1.end() /*InputIterator1*/,
      h_A2.begin() /*InputIterator2*/, h_A2.end() /*InputIterator2*/,
      h_result.begin() /*OutputIterator*/);
  thrust::set_symmetric_difference(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > &*/,
      h_A1.begin() /*InputIterator1*/, h_A1.end() /*InputIterator1*/,
      h_A2.begin() /*InputIterator2*/, h_A2.end() /*InputIterator2*/,
      h_result.begin(), thrust::less<int>());
  thrust::set_symmetric_difference(
      h_A1.begin() /*InputIterator1*/, h_A1.end() /*InputIterator1*/,
      h_A2.begin() /*InputIterator2*/, h_A2.end() /*InputIterator2*/,
      h_result.begin() /*OutputIterator*/, thrust::less<int>());
  // End
}
