// UNSUPPORTED: cuda-8.0, cuda-12.4, cuda-12.5, cuda-12.6
// UNSUPPORTED: v8.0, v12.4, v12.5, v12.6
// RUN: dpct -out-root %T/thrust_template_specific %s --cuda-include-path="%cuda-path/include" -- -ferror-limit=50
// RUN: FileCheck --input-file %T/thrust_template_specific/thrust_template_specific.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/thrust_template_specific/thrust_template_specific.dp.cpp -o %T/thrust_template_specific/thrust_template_specific.dp.o %}

#include <thrust/device_vector.h>
#include <thrust/tuple.h>
#include <thrust/iterator/constant_iterator.h>

template <typename Iterator>
void foo2() {
  // CHECK: typedef typename std::tuple_element_t<0, typename Iterator::value_type> Type;
  typedef typename Iterator::value_type :: head_type Type;
}

void foo3() {
  foo2<thrust::constant_iterator<thrust::tuple<float, double>>>();
}
