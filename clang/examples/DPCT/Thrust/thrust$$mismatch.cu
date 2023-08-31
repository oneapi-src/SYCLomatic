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
  int A[N] = {0, 5, 3, 7};
  int B[N] = {0, 5, 8, 7};

  thrust::host_vector<int> VA(A, A + N);
  thrust::host_vector<int> VB(B, B + N);
  thrust::device_vector<int> d_VA(A, A + N);
  thrust::device_vector<int> d_VB(B, B + N);

  // Start
  /*1*/ thrust::mismatch(thrust::host /*const thrust::detail::execution_policy_base<
                                   DerivedPolicy > &*/,
                   VA.begin() /*InputIterator1*/, VA.end() /*InputIterator1*/,
                   VB.begin() /*InputIterator2*/);
  /*2*/ thrust::mismatch(VA.begin() /*InputIterator1*/, VA.end() /*InputIterator1*/,
                   VB.begin() /*InputIterator2*/);
  /*3*/ thrust::mismatch(thrust::host /*const thrust::detail::execution_policy_base<
                                   DerivedPolicy > &*/,
                   VA.begin() /*InputIterator1*/, VA.end() /*InputIterator1*/,
                   VB.begin() /*InputIterator2*/,
                   thrust::equal_to<int>() /*BinaryPredicate*/);
  /*4*/ thrust::mismatch(VA.begin() /*InputIterator1*/, VA.end() /*InputIterator1*/,
                   VB.begin() /*InputIterator2*/,
                   thrust::equal_to<int>() /*BinaryPredicate*/);
  /*5*/ thrust::mismatch(thrust::device /*const thrust::detail::execution_policy_base<
                                     DerivedPolicy > &*/,
                   d_VA.begin() /*InputIterator1*/,
                   d_VA.end() /*InputIterator1*/,
                   d_VB.begin() /*InputIterator2*/);
  /*6*/ thrust::mismatch(d_VA.begin() /*InputIterator1*/,
                   d_VA.end() /*InputIterator1*/,
                   d_VB.begin() /*InputIterator2*/);
  /*7*/ thrust::mismatch(thrust::device /*const thrust::detail::execution_policy_base<
                                     DerivedPolicy > &*/,
                   d_VA.begin() /*InputIterator1*/,
                   d_VA.end() /*InputIterator1*/,
                   d_VB.begin() /*InputIterator2*/,
                   thrust::equal_to<int>() /*BinaryPredicate*/);
  /*8*/ thrust::mismatch(d_VA.begin() /*InputIterator1*/,
                   d_VA.end() /*InputIterator1*/,
                   d_VB.begin() /*InputIterator2*/,
                   thrust::equal_to<int>() /*BinaryPredicate*/);
  /*9*/ thrust::mismatch(thrust::host /*const thrust::detail::execution_policy_base<
                                   DerivedPolicy > &*/,
                   A /*InputIterator1*/, A + N /*InputIterator1*/,
                   B /*InputIterator2*/);
  /*10*/ thrust::mismatch(A /*InputIterator1*/, A + N /*InputIterator1*/,
                   B /*InputIterator2*/);
  /*11*/ thrust::mismatch(thrust::host /*const thrust::detail::execution_policy_base<
                                   DerivedPolicy > &*/,
                   A /*InputIterator1*/, A + N /*InputIterator1*/,
                   B /*InputIterator2*/,
                   thrust::equal_to<int>() /*BinaryPredicate*/);
  /*12*/ thrust::mismatch(A /*InputIterator1*/, A + N /*InputIterator1*/,
                   B /*InputIterator2*/,
                   thrust::equal_to<int>() /*BinaryPredicate*/);
  // End
}
