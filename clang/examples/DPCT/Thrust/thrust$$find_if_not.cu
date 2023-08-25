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

struct greater_than_four {
  __host__ __device__ bool operator()(int x) const { return x > 4; }
};

void test() {
  const int N = 4;
  int data[4] = {0,5, 3, 7};
  thrust::device_vector<int> device_data(data, data + N);
  thrust::host_vector<int> host_data(data, data + N);

  // Start
  thrust::find_if_not(data /*InputIterator*/, data + 3 /*InputIterator*/,
                      greater_than_four() /*Predicate*/);
  thrust::find_if_not(device_data.begin() /*InputIterator*/,
                      device_data.end() /*InputIterator*/,
                      greater_than_four() /*Predicate*/);
  thrust::find_if_not(host_data.begin() /*InputIterator*/,
                      host_data.end() /*InputIterator*/,
                      greater_than_four() /*Predicate*/);
  thrust::find_if_not(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > &*/,
      data /*InputIterator*/, data + 3 /*InputIterator*/,
      greater_than_four() /*Predicate*/);
  thrust::find_if_not(
      thrust::device /*const thrust::detail::execution_policy_base<
                        DerivedPolicy > &*/,
      device_data.begin() /*InputIterator*/,
      device_data.end() /*InputIterator*/, greater_than_four() /*Predicate*/);
  thrust::find_if_not(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > &*/,
      host_data.begin() /*InputIterator*/, host_data.end() /*InputIterator*/,
      greater_than_four() /*Predicate*/);
  // End
}
