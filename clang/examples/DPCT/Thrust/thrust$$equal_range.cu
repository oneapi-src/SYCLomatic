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

struct Int {
  __host__ __device__ Int(int x) : val(x) {}
  int val;
};

void test() {
  int data[] = {0, 2, 5, 7, 8};
  const int N = 5;
  thrust::host_vector<int> host_vec(data, data + N);
  thrust::device_vector<int> device_vec(data, data + N);

  // Start
  thrust::equal_range(
      thrust::device /*const thrust::detail::execution_policy_base<
                        DerivedPolicy > &*/,
      device_vec.begin() /*ForwardIterator*/,
      device_vec.end() /*ForwardIterator*/, 0 /*const LessThanComparable &*/);
  thrust::equal_range(device_vec.begin() /*ForwardIterator*/,
                      device_vec.end() /*ForwardIterator*/,
                      0 /*const LessThanComparable &*/);
  thrust::equal_range(
      thrust::device /*const thrust::detail::execution_policy_base<
                        DerivedPolicy > &*/,
      device_vec.begin() /*ForwardIterator*/,
      device_vec.end() /*ForwardIterator*/, 0 /*const T &*/,
      thrust::less<int>() /*StrictWeakOrdering */);
  thrust::equal_range(device_vec.begin() /*ForwardIterator*/,
                      device_vec.end() /*ForwardIterator*/,
                      0 /*const LessThanComparable &*/, thrust::less<int>());
  thrust::equal_range(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > &*/,
      host_vec.begin() /*ForwardIterator*/, host_vec.end() /*ForwardIterator*/,
      0 /*const LessThanComparable &*/);
  thrust::equal_range(host_vec.begin() /*ForwardIterator*/,
                      host_vec.end() /*ForwardIterator*/,
                      0 /*const LessThanComparable &*/);
  thrust::equal_range(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > &*/,
      host_vec.begin() /*ForwardIterator*/, host_vec.end() /*ForwardIterator*/,
      0 /*const T &*/, thrust::less<int>() /*StrictWeakOrdering */);
  thrust::equal_range(host_vec.begin(), host_vec.end(), 0 /*const T &*/,
                      thrust::less<int>() /*StrictWeakOrdering */);
  thrust::equal_range(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > &*/,
      data /*ForwardIterator*/, data + N /*ForwardIterator*/,
      0 /*const LessThanComparable &*/);
  thrust::equal_range(data /*ForwardIterator*/, data + N /*ForwardIterator*/,
                      0 /*const LessThanComparable &*/);
  thrust::equal_range(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > &*/,
      data /*ForwardIterator*/, data + N /*ForwardIterator*/,
      0 /*const LessThanComparable &*/,
      thrust::less<int>() /*StrictWeakOrdering */);
  thrust::equal_range(data /*ForwardIterator*/, data + N /*ForwardIterator*/,
                      0 /*const T &*/,
                      thrust::less<int>() /*StrictWeakOrdering */);
  // End
}
