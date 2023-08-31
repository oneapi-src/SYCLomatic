#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/set_operations.h>

void test() {

  thrust::device_vector<int> d_v1(2), d_v2(2);
  thrust::host_vector<int> h_v1(2), h_v2(2);
  int v1[2], v2[2];

  // Start
  /*1*/ thrust::swap_ranges(
      thrust::device /*const thrust::detail::execution_policy_base<
                        DerivedPolicy > & 	exec*/,
      d_v1.begin()/*ForwardIterator1*/, d_v1.end()/*ForwardIterator1*/, d_v2.begin()/*ForwardIterator2*/);
  /*2*/ thrust::swap_ranges(d_v1.begin(), d_v1.end(), d_v2.begin()/*ForwardIterator2*/);
  /*3*/ thrust::swap_ranges(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > & 	exec*/,
      h_v1.begin()/*ForwardIterator1*/, h_v1.end()/*ForwardIterator1*/, h_v2.begin()/*ForwardIterator2*/);
  /*4*/ thrust::swap_ranges(h_v1.begin()/*ForwardIterator1*/, h_v1.end()/*ForwardIterator1*/, h_v2.begin()/*ForwardIterator2*/);
  /*5*/ thrust::swap_ranges(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > & 	exec*/,
      v1/*ForwardIterator1*/, v1 + 2/*ForwardIterator1*/, v2/*ForwardIterator2*/);
  /*6*/ thrust::swap_ranges(v1 /*ForwardIterator1*/, v1 + 2 /*ForwardIterator1*/,
                      v2 /*ForwardIterator2*/);
  // End
}
