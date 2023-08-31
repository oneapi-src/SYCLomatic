#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/find.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/mismatch.h>
#include <thrust/partition.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/replace.h>
#include <thrust/reverse.h>
#include <thrust/scatter.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/transform_scan.h>
#include <thrust/uninitialized_copy.h>
#include <thrust/unique.h>
#include <vector>

struct is_even
{
  __host__ __device__
  bool operator()(const int &x)
  {
    return (x % 2) == 0;
  }
};

void test() {
  int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int S[] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  int result[10];
  const int N = sizeof(data) / sizeof(int);
  int *evens = result;
  int *odds = result + N / 2;

  thrust::host_vector<int> host_a(data, data + N);
  thrust::host_vector<int> host_evens(N / 2);
  thrust::host_vector<int> host_odds(N / 2);

  thrust::device_vector<int> device_a(data, data + N);
  thrust::device_vector<int> device_evens(N / 2);
  thrust::device_vector<int> device_odds(N / 2);
  thrust::host_vector<int> host_S(S, S + N);
  thrust::device_vector<int> device_S(S, S + N);

  // Start
  /*1*/ thrust::stable_partition_copy(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > &*/,
      data /*InputIterator*/, data + N /*InputIterator*/,
      evens /*OutputIterator1*/, odds /*OutputIterator2*/,
      is_even() /*Predicate*/);
  /*2*/ thrust::stable_partition_copy(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > &*/,
      host_a.begin() /*InputIterator*/, host_a.begin() + N /*InputIterator*/,
      host_evens.begin() /*OutputIterator1*/,
      host_odds.begin() /*OutputIterator2*/, is_even() /*Predicate*/);
  /*3*/ thrust::stable_partition_copy(
      thrust::device /*const thrust::detail::execution_policy_base<
                        DerivedPolicy > &*/,
      device_a.begin() /*InputIterator*/,
      device_a.begin() + N /*InputIterator*/,
      device_evens.begin() /*OutputIterator2*/,
      device_odds.begin() /*OutputIterator2*/, is_even() /*Predicate*/);
  /*4*/ thrust::stable_partition_copy(
      data /*InputIterator*/, data + N /*InputIterator*/,
      evens /*OutputIterator1*/, odds /*OutputIterator2*/,
      is_even() /*Predicate*/);
  /*5*/ thrust::stable_partition_copy(
      host_a.begin() /*InputIterator*/, host_a.begin() + N /*InputIterator*/,
      host_evens.begin() /*OutputIterator1*/,
      host_odds.begin() /*OutputIterator2*/, is_even());
  /*6*/ thrust::stable_partition_copy(device_a.begin() /*InputIterator*/,
                                device_a.begin() + N /*InputIterator*/,
                                device_evens.begin() /*OutputIterator1*/,
                                device_odds.begin() /*OutputIterator2*/,
                                is_even() /*Predicate*/);
  /*7*/ thrust::stable_partition_copy(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > &*/,
      data /*InputIterator1*/, data + N /*InputIterator1*/,
      S /*InputIterator2*/, evens /*OutputIterator1*/, odds /*OutputIterator2*/,
      is_even() /*Predicate*/);
  /*8*/ thrust::stable_partition_copy(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > &*/,
      host_a.begin() /*InputIterator1*/, host_a.begin() + N /*InputIterator1*/,
      host_S.begin() /*InputIterator2*/, host_evens.begin() /*OutputIterator1*/,
      host_odds.begin() /*OutputIterator2*/, is_even() /*Predicate*/);
  /*9*/ thrust::stable_partition_copy(
      thrust::device /*const thrust::detail::execution_policy_base<
                        DerivedPolicy > &*/,
      device_a.begin() /*InputIterator1*/,
      device_a.begin() + N /*InputIterator1*/,
      device_S.begin() /*InputIterator2*/,
      device_evens.begin() /*OutputIterator1*/,
      device_odds.begin() /*OutputIterator2*/, is_even() /*Predicate*/);
  /*10*/ thrust::stable_partition_copy(
      data /*InputIterator1*/, data + N /*InputIterator1*/,
      S /*InputIterator2*/, evens /*OutputIterator1*/, odds /*OutputIterator2*/,
      is_even() /*Predicate*/);
  /*11*/ thrust::stable_partition_copy(
      host_a.begin() /*InputIterator1*/, host_a.begin() + N /*InputIterator1*/,
      host_S.begin() /*InputIterator2*/, host_evens.begin() /*OutputIterator1*/,
      host_odds.begin() /*OutputIterator2*/, is_even() /*Predicate*/);
  /*12*/ thrust::stable_partition_copy(device_a.begin() /*InputIterator1*/,
                                device_a.begin() + N /*InputIterator1*/,
                                device_S.begin() /*InputIterator2*/,
                                device_evens.begin() /*OutputIterator1*/,
                                device_odds.begin() /*OutputIterator2*/,
                                is_even() /*Predicate*/);
  // End
}
