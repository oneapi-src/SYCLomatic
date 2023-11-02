// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct -out-root %T/thrust/thrust_testing %s --cuda-include-path="%cuda-path/include" -extra-arg-before="-I%S" -- -x cuda --cuda-host-only -std=c++17
// RUN: FileCheck --input-file %T//thrust/thrust_testing/foo.dp.cpp --match-full-lines %s

#include <algorithm>
#include <thrust/complex.h> // here complex.h is user defined head file
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#include "../foo.h"

// CHECK: void foo() { int min = std::min(1, 2); }
void foo() { int min = THRUST_NS_QUALIFIER::min<int>(1, 2); }

// uninitialized template function baz() is used to check the crash issue when SYCLomatic parses it
template <typename ForwardIterator1, typename ForwardIterator2>
void baz(ForwardIterator1 first1, ForwardIterator1 last1,
         ForwardIterator2 first2, ForwardIterator2 last2) {
  typedef
      typename THRUST_NS_QUALIFIER::iterator_difference<ForwardIterator1>::type
          difference_type;
  typedef typename THRUST_NS_QUALIFIER::iterator_value<ForwardIterator1>::type
      InputType;

  difference_type length1 = THRUST_NS_QUALIFIER::distance(first1, last1);
  difference_type length2 = THRUST_NS_QUALIFIER::distance(first2, last2);
  difference_type min_length = THRUST_NS_QUALIFIER::min(length1, length2);
}

int main() {

  // CHECK: sycl::range<3> t(1, 1, 1);
  dim3 t;
  return 0;
}
