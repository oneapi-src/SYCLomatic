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
  int data[6] = {1, 0, 2, 2, 1, 3};
  thrust::negate<int> unary_op;
  thrust::plus<int> binary_op;
  thrust::host_vector<int> h_vec_data(data, data + N);
  thrust::device_vector<int> d_vec_data(data, data + N);

  // Start
  thrust::transform_inclusive_scan(
      data /*InputIterator*/, data + N /*InputIterator*/,
      data /*OutputIterator*/, unary_op /*UnaryFunction*/,
      binary_op /*AssociativeOperator*/);
  thrust::transform_inclusive_scan(
      h_vec_data.begin() /*InputIterator*/, h_vec_data.end() /*InputIterator*/,
      h_vec_data.begin() /*OutputIterator*/, unary_op /*UnaryFunction*/,
      binary_op /*AssociativeOperator*/);
  thrust::transform_inclusive_scan(
      d_vec_data.begin() /*InputIterator*/, d_vec_data.end() /*InputIterator*/,
      d_vec_data.begin() /*OutputIterator*/, unary_op /*UnaryFunction*/,
      binary_op /*AssociativeOperator*/);
  thrust::transform_inclusive_scan(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > &*/,
      data /*InputIterator*/, data + N /*InputIterator*/,
      data /*OutputIterator*/, unary_op /*UnaryFunction*/,
      binary_op /*AssociativeOperator*/);
  thrust::transform_inclusive_scan(
      thrust::host /*const thrust::detail::execution_policy_base< DerivedPolicy
                      > &*/,
      h_vec_data.begin() /*InputIterator*/, h_vec_data.end() /*InputIterator*/,
      h_vec_data.begin() /*OutputIterator*/, unary_op /*UnaryFunction*/,
      binary_op /*AssociativeOperator*/);
  thrust::transform_inclusive_scan(
      thrust::device /*const thrust::detail::execution_policy_base<
                        DerivedPolicy > &*/,
      d_vec_data.begin() /*InputIterator*/, d_vec_data.end() /*InputIterator*/,
      d_vec_data.begin() /*OutputIterator*/, unary_op /*UnaryFunction*/,
      binary_op /*AssociativeOperator*/);
  // End
}
