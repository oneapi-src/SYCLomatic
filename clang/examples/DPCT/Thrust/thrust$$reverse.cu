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
  const int N = 6;
  int data[N] = {0, 1, 2, 3, 4, 5};
  thrust::device_vector<int> device_data(data, data + N);
  thrust::host_vector<int> host_data(data, data + N);

  // Start
  thrust::reverse(thrust::device /*const thrust::detail::execution_policy_base<
                                    DerivedPolicy > &*/
                  ,
                  device_data.begin() /*BidirectionalIterator*/,
                  device_data.end() /*BidirectionalIterator*/);
  thrust::reverse(thrust::host /*const thrust::detail::execution_policy_base<
                                  DerivedPolicy > &*/
                  ,
                  host_data.begin() /*BidirectionalIterator*/,
                  host_data.end() /*BidirectionalIterator*/);
  thrust::reverse(thrust::host /*const thrust::detail::execution_policy_base<
                                  DerivedPolicy > &*/
                  ,
                  data /*BidirectionalIterator*/,
                  data + N /*BidirectionalIterator*/);
  thrust::reverse(device_data.begin() /*BidirectionalIterator*/,
                  device_data.end() /*BidirectionalIterator*/);
  thrust::reverse(host_data.begin() /*BidirectionalIterator*/,
                  host_data.end() /*BidirectionalIterator*/);
  thrust::reverse(data /*BidirectionalIterator*/,
                  data + N /*BidirectionalIterator*/);
  // End
}
