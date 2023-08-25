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
  const int N = 137;
  Int val(46);
  thrust::device_ptr<Int> array = thrust::device_malloc<Int>(N);
  thrust::device_vector<Int> d_input(N, val);
  thrust::device_ptr<Int> d_array = thrust::device_malloc<Int>(N);
  int h_data[N];
  int h_array[N];

  // Start
  thrust::uninitialized_copy_n(d_input.begin() /*InputIterator*/, N /*Size*/,
                               d_array /*ForwardIterator*/);
  thrust::uninitialized_copy_n(h_data /*InputIterator*/, N /*Size*/,
                               h_array /*ForwardIterator*/);
  thrust::uninitialized_copy_n(
      thrust::device /*const thrust::detail::execution_policy_base<
                        DerivedPolicy > &*/
      ,
      d_input.begin() /*InputIterator*/, N /*Size*/,
      d_array /*ForwardIterator*/);
  thrust::uninitialized_copy_n(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > &*/
      ,
      h_data /*InputIterator*/, N /*Size*/, h_array /*ForwardIterator*/);
  // End
}
