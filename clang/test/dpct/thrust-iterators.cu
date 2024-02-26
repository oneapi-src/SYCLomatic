// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0
// RUN: dpct --format-range=none -out-root %T/thrust-iterators %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/thrust-iterators/thrust-iterators.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/thrust-iterators/thrust-iterators.dp.cpp -o %T/thrust-iterators/thrust-iterators.dp.o %}
// CHECK: #include <oneapi/dpl/execution>
// CHECK-NEXT: #include <oneapi/dpl/algorithm>
// CHECK-EMPTY:
// CHECK-NEXT: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <dpct/dpl_utils.hpp>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>

template <typename I, typename F>
class StridedRange {
public:
  typedef typename thrust::iterator_difference<I>::type difference_type;
// CHECK:  typedef typename oneapi::dpl::counting_iterator<difference_type> CI;
  typedef typename thrust::counting_iterator<difference_type> CI;
// CHECK:  typedef typename oneapi::dpl::permutation_iterator<I, I> PI;
  typedef typename thrust::permutation_iterator<I, I> PI;

// CHECK:  typedef typename oneapi::dpl::transform_iterator<CI, F> TI1;
  typedef typename thrust::transform_iterator<F, CI> TI1;
// CHECK:  typedef typename oneapi::dpl::transform_iterator<oneapi::dpl::counting_iterator<int>, F> TI2;
  typedef typename thrust::transform_iterator<F, thrust::counting_iterator<int>> TI2;
// CHECK:  typedef typename oneapi::dpl::transform_iterator<oneapi::dpl::transform_iterator<I, F>, F> TI3;
  typedef typename thrust::transform_iterator<F, thrust::transform_iterator<F, I>> TI3;

// CHECK:  oneapi::dpl::counting_iterator<difference_type> cIt;
  thrust::counting_iterator<difference_type> cIt;
// CHECK:  oneapi::dpl::permutation_iterator<I, TI1> pIt;
  thrust::permutation_iterator<I, TI1> pIt;
// CHECK:  oneapi::dpl::transform_iterator<F, CI>  tIt;
  thrust::transform_iterator<CI, F>  tIt;
};


int main() {
  thrust::device_vector<int> d_vec(10);
// CHECK: auto iter = oneapi::dpl::make_reverse_iterator(d_vec.begin());
  auto iter = thrust::make_reverse_iterator(d_vec.begin());
  return 0;
}
