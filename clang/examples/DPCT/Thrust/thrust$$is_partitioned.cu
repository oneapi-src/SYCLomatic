#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
#include <thrust/partition.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>

struct is_even {
  __host__ __device__ bool operator()(const int &x) const {
    return (x % 2) == 0;
  }
};

void test() {

  int A[] = {2, 4, 6, 8, 10, 1, 3, 5, 7, 9};
  bool result;
  thrust::host_vector<int> h_A(A, A + 10);
  thrust::device_vector<int> d_A(A, A + 10);

  // Start
  /*1*/ thrust::is_partitioned(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > & 	exec*/,
      A, A + 10, is_even());
  /*2*/ thrust::is_partitioned(A /*InputIterator */, A + 10 /*InputIterator */,
                         is_even() /*Predicate*/);
  /*3*/ thrust::is_partitioned(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > & 	exec*/,
      h_A.begin() /*InputIterator */, h_A.end() /*InputIterator */, is_even());
  /*4*/ thrust::is_partitioned(h_A.begin() /*InputIterator */,
                         h_A.end() /*InputIterator */, is_even() /*Predicate*/);
  /*5*/ thrust::is_partitioned(
      thrust::device /*const thrust::detail::execution_policy_base<
                        DerivedPolicy > & 	exec*/,
      d_A.begin() /*InputIterator */, d_A.end() /*InputIterator */, is_even());
  /*6*/ thrust::is_partitioned(d_A.begin() /*InputIterator */,
                         d_A.end() /*InputIterator */, is_even() /*Predicate*/);
  // End
}
