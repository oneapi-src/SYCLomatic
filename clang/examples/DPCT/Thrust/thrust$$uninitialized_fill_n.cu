#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/set_operations.h>

struct Int {
  __host__ __device__ Int(int x) : val(x) {}
  int val;
};

void test() {

  const int N = 137;
  Int int_val(46);
  int val(46);
  thrust::device_ptr<Int> d_array = thrust::device_malloc<Int>(N);
  int data[N];

  // Start
  thrust::uninitialized_fill_n(d_array /*ForwardIterator */, N /*Size */,
                               int_val /*	const T & */);
  thrust::uninitialized_fill_n(
      thrust::device /*const thrust::detail::execution_policy_base<
                        DerivedPolicy > &*/
      ,
      d_array /*ForwardIterator */, N /*Size */, int_val /*	const T & */);
  thrust::uninitialized_fill_n(data /*ForwardIterator */, N /*Size */,
                               val /*	const T & */);
  thrust::uninitialized_fill_n(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > &*/
      ,
      data /*ForwardIterator */, N /*Size */, val /*	const T & */);
  // End
}
