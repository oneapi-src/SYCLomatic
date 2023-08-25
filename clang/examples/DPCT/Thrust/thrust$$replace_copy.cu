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

void test() {
  const int N = 4;
  int data[] = {1, 2, 3, 1};
  thrust::device_vector<int> d_data(data, data +N);
  thrust::device_vector<int> d_result(4);
  thrust::host_vector<int> h_data(data, data +N);
  thrust::host_vector<int> h_result(4);
  int result[N];

  // Start
  thrust::replace_copy(
      thrust::device /*const thrust::detail::execution_policy_base<
                        DerivedPolicy > &*/,
      d_data.begin() /*InputIterator*/, d_data.end() /*InputIterator*/,
      d_result.begin() /*OutputIterator*/, 1 /*const T &*/, 99 /*const T &*/);
  thrust::replace_copy(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > &*/,
      h_data.begin() /*InputIterator*/, h_data.end() /*InputIterator*/,
      h_result.begin() /*OutputIterator*/, 1 /*const T &*/, 99 /*const T &*/);
  thrust::replace_copy(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > &*/,
      data /*InputIterator*/, data + N /*InputIterator*/,
      result /*OutputIterator*/, 1 /*const T &*/, 99 /*const T &*/);
  thrust::replace_copy(
      d_data.begin() /*InputIterator*/, d_data.end() /*InputIterator*/,
      d_result.begin() /*OutputIterator*/, 1 /*const T &*/, 99 /*const T &*/);
  thrust::replace_copy(
      h_data.begin() /*InputIterator*/, h_data.end() /*InputIterator*/,
      h_result.begin() /*OutputIterator*/, 1 /*const T &*/, 99 /*const T &*/);
  thrust::replace_copy(data /*InputIterator*/, data + N /*InputIterator*/,
                       result /*OutputIterator*/, 1 /*const T &*/,
                       99 /*const T &*/);
  // End
}
