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

  bool A[3] = {true, true, false};
  thrust::host_vector<bool> h_A(A, A + 3);
  thrust::device_vector<bool> d_A(A, A + 3);
  bool result;

  // Start
  /*1*/ thrust::none_of(thrust::host /*const thrust::detail::execution_policy_base<
                                  DerivedPolicy > &*/,
                  A /*InputIterator */, A + 2 /*InputIterator*/,
                  thrust::identity<bool>() /*Predicate*/);
  /*2*/ thrust::none_of(A /*InputIterator */, A + 2 /*InputIterator*/,
                  thrust::identity<bool>() /*Predicate*/);
  /*3*/ thrust::none_of(thrust::host /*const thrust::detail::execution_policy_base<
                                  DerivedPolicy > &*/,
                  h_A.begin() /*InputIterator*/,
                  h_A.begin() + 2 /*InputIterator*/,
                  thrust::identity<bool>() /*Predicate*/);
  /*4*/ thrust::none_of(h_A.begin() /*InputIterator*/,
                  h_A.begin() + 2 /*InputIterator*/,
                  thrust::identity<bool>() /*Predicate*/);
  /*5*/ thrust::none_of(thrust::device /*const thrust::detail::execution_policy_base<
                                    DerivedPolicy > &*/,
                  d_A.begin() /*InputIterator*/,
                  d_A.begin() + 2 /*InputIterator */,
                  thrust::identity<bool>() /*Predicate*/);
  /*6*/ thrust::none_of(d_A.begin() /*InputIterator*/,
                  d_A.begin() + 2 /*InputIterator*/,
                  thrust::identity<bool>() /*Predicate*/);
  // End
}
