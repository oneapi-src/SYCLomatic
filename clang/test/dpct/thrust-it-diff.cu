// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0
// RUN: dpct --format-range=none -out-root %T/thrust-it-diff %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/thrust-it-diff/thrust-it-diff.dp.cpp --match-full-lines %s
// CHECK: #include <oneapi/dpl/execution>
// CHECK-NEXT: #include <oneapi/dpl/algorithm>
// CHECK-NEXT: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <dpct/dpl_utils.hpp>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>

template <typename Iterator>
class C {
public:
// CHECK:  typedef typename std::iterator_traits<Iterator> IteratorDiff;
  typedef typename thrust::iterator_difference<Iterator> IteratorDiff;
// CHECK:  typedef typename std::iterator_traits<Iterator>::difference_type difference_type;
  typedef typename thrust::iterator_difference<Iterator>::type difference_type;
// CHECK:  typename std::iterator_traits<Iterator>::difference_type IDTFieldDecl;
  typename thrust::iterator_difference<Iterator>::type IDTFieldDecl;
};

// TODO:
typename thrust::iterator_difference<int *>::type IDTVarDecl;

