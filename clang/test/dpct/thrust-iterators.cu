// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0
// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/thrust-iterators.dp.cpp --match-full-lines %s
// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <dpct/dpstd_utils.hpp>
// CHECK-NEXT: #include <dpstd/execution>
// CHECK-NEXT: #include <dpstd/algorithm>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>

template <typename Iterator, typename StrideFunctor>
class StridedRange {
public:
  // TODO
  typedef typename thrust::iterator_difference<Iterator>::type difference_type;
// CHECK:  typedef typename dpstd::counting_iterator<difference_type>                   CountingIterator;
  typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
// CHECK:  typedef typename dpstd::transform_iterator<StrideFunctor, CountingIterator>  TransformIterator;
  typedef typename thrust::transform_iterator<StrideFunctor, CountingIterator>  TransformIterator;
// CHECK:  typedef typename dpstd::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;
  typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;

// CHECK:  dpstd::counting_iterator<difference_type> cIt;
  thrust::counting_iterator<difference_type> cIt;
// CHECK:  dpstd::transform_iterator<StrideFunctor, CountingIterator>  tIt;
  thrust::transform_iterator<StrideFunctor, CountingIterator>  tIt;
// CHECK:  dpstd::permutation_iterator<Iterator,TransformIterator> pIt;
  thrust::permutation_iterator<Iterator,TransformIterator> pIt;
};
