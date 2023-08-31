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

struct compare_modulo_two {
  __host__ __device__ bool operator()(int x, int y) const {
    return (x % 2) == (y % 2);
  }
};

void test() {
  const int N = 7;

  int A1[N] = {3, 1, 4, 1, 5, 9, 3};
  int A2[N] = {3, 1, 4, 2, 8, 5, 7};
  int x[N] = {0, 2, 4, 6, 8, 10};
  int y[N] = {1, 3, 5, 7, 9, 11};
  thrust::host_vector<int> h_A1(A1, A1 + N);
  thrust::host_vector<int> h_A2(A2, A2 + N);
  thrust::host_vector<int> h_x(x, x + N);
  thrust::host_vector<int> h_y(y, y + N);
  thrust::device_vector<int> d_A1(A1, A1 + N);
  thrust::device_vector<int> d_A2(A2, A2 + N);
  thrust::device_vector<int> d_x(x, x + N);
  thrust::device_vector<int> d_y(y, y + N);

  // Start
  /*1*/ thrust::equal(thrust::host /*const thrust::detail::execution_policy_base<
                                DerivedPolicy > &*/,
                A1 /*InputIterator1*/, A1 + N /*InputIterator1*/,
                A2 /*InputIterator2*/);
  /*2*/ thrust::equal(A1 /*InputIterator1*/, A1 + N /*InputIterator1*/,
                A2 /*InputIterator2*/);
  /*3*/ thrust::equal(x /*InputIterator1*/, x + N /*InputIterator1*/, y,
                compare_modulo_two() /*BinaryPredicate*/);
  /*4*/ thrust::equal(thrust::host /*const thrust::detail::execution_policy_base<
                                DerivedPolicy > &*/,
                x /*InputIterator1*/, x + N /*InputIterator1*/, y,
                compare_modulo_two() /*BinaryPredicate*/);
  /*5*/ thrust::equal(thrust::host /*const thrust::detail::execution_policy_base<
                                DerivedPolicy > &*/,
                h_A1.begin() /*InputIterator1*/, h_A1.end() /*InputIterator1*/,
                h_A2.begin() /*InputIterator2*/);
  /*6*/ thrust::equal(h_A1.begin() /*InputIterator1*/, h_A1.end() /*InputIterator1*/,
                h_A2.begin() /*InputIterator2*/);
  /*7*/ thrust::equal(h_x.begin() /*InputIterator1*/, h_x.end() /*InputIterator1*/,
                h_y.begin() /*InputIterator2*/,
                compare_modulo_two() /*BinaryPredicate*/);
  /*8*/ thrust::equal(thrust::host /*const thrust::detail::execution_policy_base<
                                DerivedPolicy > &*/,
                h_x.begin() /*InputIterator1*/, h_x.end() /*InputIterator1*/,
                h_y.begin() /*InputIterator2*/,
                compare_modulo_two() /*BinaryPredicate*/);
  /*9*/ thrust::equal(thrust::device /*const thrust::detail::execution_policy_base<
                                  DerivedPolicy > &*/,
                d_A1.begin() /*InputIterator1*/, d_A1.end() /*InputIterator1*/,
                d_A2.begin());
  /*10*/ thrust::equal(d_A1.begin() /*InputIterator1*/, d_A1.end() /*InputIterator1*/,
                d_A2.begin() /*InputIterator2*/);
  /*11*/ thrust::equal(d_x.begin() /*InputIterator1*/, d_x.end() /*InputIterator1*/,
                d_y.begin() /*InputIterator2*/,
                compare_modulo_two() /*BinaryPredicate*/);
  /*12*/ thrust::equal(thrust::device /*const thrust::detail::execution_policy_base<
                                  DerivedPolicy > &*/,
                d_x.begin() /*InputIterator1*/, d_x.end() /*InputIterator1*/,
                d_y.begin() /*InputIterator2*/,
                compare_modulo_two() /*BinaryPredicate*/);
  // End
}
