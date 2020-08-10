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

template <typename I, typename F>
class StridedRange {
public:
  typedef typename thrust::iterator_difference<I>::type difference_type;
// CHECK:  typedef typename dpstd::counting_iterator<difference_type> CI;
  typedef typename thrust::counting_iterator<difference_type> CI;
// CHECK:  typedef typename dpstd::permutation_iterator<I, I> PI;
  typedef typename thrust::permutation_iterator<I, I> PI;

// CHECK:  typedef typename dpstd::transform_iterator<CI, F> TI1;
  typedef typename thrust::transform_iterator<F, CI> TI1;
// CHECK:  typedef typename dpstd::transform_iterator<dpstd::counting_iterator<int>, F> TI2;
  typedef typename thrust::transform_iterator<F, thrust::counting_iterator<int>> TI2;
// CHECK:  typedef typename dpstd::transform_iterator<dpstd::transform_iterator<I, F>, F> TI3;
  typedef typename thrust::transform_iterator<F, thrust::transform_iterator<F, I>> TI3;

// CHECK:  dpstd::counting_iterator<difference_type> cIt;
  thrust::counting_iterator<difference_type> cIt;
// CHECK:  dpstd::permutation_iterator<I, TI1> pIt;
  thrust::permutation_iterator<I, TI1> pIt;
// CHECK:  dpstd::transform_iterator<F, CI>  tIt;
  thrust::transform_iterator<CI, F>  tIt;
};
