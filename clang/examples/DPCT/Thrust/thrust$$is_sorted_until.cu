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

  int A[8] = {0, 1, 2, 3, 0, 1, 2, 3};
  thrust::host_vector<int> h_A(A, A + 8);
  thrust::device_vector<int> d_A(A, A + 8);
  thrust::greater<int> comp;
  int *B;
  thrust::host_vector<int>::iterator h_end;
  thrust::device_vector<int>::iterator d_end;
  // Start
  /*1*/ thrust::is_sorted_until(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > &*/,
      A /*ForwardIterator */, A + 8 /*ForwardIterator */);
  /*2*/ thrust::is_sorted_until(A /*ForwardIterator */, A + 8 /*ForwardIterator */);
  /*3*/ thrust::is_sorted_until(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > &*/,
      A /*ForwardIterator */, A + 8 /*ForwardIterator */, comp /*Compare */);
  /*4*/ thrust::is_sorted_until(A /*ForwardIterator */, A + 8 /*ForwardIterator */,
                          comp /*Compare */);
  /*5*/ thrust::is_sorted_until(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > &*/,
      h_A.begin() /*ForwardIterator */, h_A.end() /*ForwardIterator */);
  /*6*/ thrust::is_sorted_until(h_A.begin() /*ForwardIterator */,
                          h_A.end() /*ForwardIterator */);
  /*7*/ thrust::is_sorted_until(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > &*/,
      h_A.begin() /*ForwardIterator */, h_A.end() /*ForwardIterator */,
      comp /*Compare */);
  /*8*/ thrust::is_sorted_until(h_A.begin() /*ForwardIterator */,
                          h_A.end() /*ForwardIterator */, comp /*Compare */);
  /*9*/ thrust::is_sorted_until(
      thrust::device /*const thrust::detail::execution_policy_base<
                        DerivedPolicy > &*/,
      d_A.begin() /*ForwardIterator */, d_A.end() /*ForwardIterator */);
  /*10*/ thrust::is_sorted_until(d_A.begin() /*ForwardIterator */,
                          d_A.end() /*ForwardIterator */);
  /*11*/ thrust::is_sorted_until(
      thrust::device /*const thrust::detail::execution_policy_base<
                        DerivedPolicy > &*/,
      d_A.begin() /*ForwardIterator */, d_A.end() /*ForwardIterator */,
      comp /*Compare */);
  /*12*/ thrust::is_sorted_until(d_A.begin() /*ForwardIterator */,
                          d_A.end() /*ForwardIterator */, comp /*Compare */);
  // End
}
