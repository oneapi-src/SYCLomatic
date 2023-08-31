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
  int S[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const int N = sizeof(data) / sizeof(int);
  thrust::host_vector<int> host_data(data, data + N);
  thrust::device_vector<int> device_data(data, data + N);
  thrust::host_vector<int> host_S(S, S + N);
  thrust::device_vector<int> device_s(S, S + N);

  // Start
  /*1*/ thrust::stable_partition(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > &*/,
      data /*ForwardIterator*/, data + N /*ForwardIterator*/, is_even());
  /*2*/ thrust::stable_partition(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > &*/,
      host_data.begin() /*ForwardIterator*/,
      host_data.begin() + N /*ForwardIterator*/, is_even());
  /*3*/ thrust::stable_partition(
      thrust::device /*const thrust::detail::execution_policy_base<
                        DerivedPolicy > &*/,
      device_data.begin() /*ForwardIterator*/,
      device_data.begin() + N /*ForwardIterator*/, is_even() /*Predicate*/);
  /*4*/ thrust::stable_partition(data /*ForwardIterator*/,
                           data + N /*ForwardIterator*/,
                           is_even() /*Predicate*/);
  /*5*/ thrust::stable_partition(host_data.begin() /*ForwardIterator*/,
                           host_data.begin() + N /*ForwardIterator*/,
                           is_even() /*Predicate*/);
  /*6*/ thrust::stable_partition(device_data.begin() /*ForwardIterator*/,
                           device_data.begin() + N /*ForwardIterator*/,
                           is_even() /*Predicate*/);
  /*7*/ thrust::stable_partition(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > &*/,
      data /*ForwardIterator*/, data + N /*ForwardIterator*/, S, is_even());
  /*8*/ thrust::stable_partition(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > &*/,
      host_data.begin() /*ForwardIterator*/,
      host_data.begin() + N /*ForwardIterator*/, host_S.begin(), is_even());
  /*9*/ thrust::stable_partition(
      thrust::device /*const thrust::detail::execution_policy_base<
                        DerivedPolicy > &*/,
      device_data.begin() /*ForwardIterator*/,
      device_data.begin() + N /*ForwardIterator*/,
      device_s.begin() /*InputIterator*/, is_even() /*Predicate*/);
  /*10*/ thrust::stable_partition(data /*ForwardIterator*/,
                           data + N /*ForwardIterator*/, S /*InputIterator*/,
                           is_even() /*Predicate*/);
  /*11*/ thrust::stable_partition(host_data.begin() /*ForwardIterator*/,
                           host_data.begin() + N /*ForwardIterator*/,
                           host_S.begin() /*InputIterator*/,
                           is_even() /*Predicate*/);
  /*12*/ thrust::stable_partition(device_data.begin() /*ForwardIterator*/,
                           device_data.begin() + N /*ForwardIterator*/,
                           device_s.begin() /*InputIterator*/,
                           is_even() /*Predicate*/);
  // End
}
