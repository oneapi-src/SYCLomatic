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
  const int N = 6;
  int data[N] = {3, 1, 4, 1, 5, 9};
  thrust::device_vector<int> device_data(data, data + N);
  thrust::host_vector<int> host_data(data, data + N);

  // Start
  /*1*/ thrust::remove(thrust::host /*const thrust::detail::execution_policy_base<
                                 DerivedPolicy > &*/,
                 data /*ForwardIterator*/, data + N /*ForwardIterator*/,
                 1 /*const T &*/);
  /*2*/ thrust::remove(thrust::host /*const thrust::detail::execution_policy_base<
                                 DerivedPolicy > &*/,
                 host_data.begin() /*ForwardIterator*/,
                 host_data.begin() + N /*ForwardIterator*/, 1 /*const T &*/);
  /*3*/ thrust::remove(thrust::device /*const thrust::detail::execution_policy_base<
                                   DerivedPolicy > &*/,
                 device_data.begin() /*ForwardIterator*/,
                 device_data.begin() + N /*ForwardIterator*/, 1 /*const T &*/);
  /*4*/ thrust::remove(data /*ForwardIterator*/, data + N /*ForwardIterator*/, 1);
  /*5*/ thrust::remove(host_data.begin() /*ForwardIterator*/,
                 host_data.begin() + N /*ForwardIterator*/, 1 /*const T &*/);
  /*6*/ thrust::remove(device_data.begin() /*ForwardIterator*/,
                 device_data.begin() + N /*ForwardIterator*/, 1 /*const T &*/);
  // End
}
