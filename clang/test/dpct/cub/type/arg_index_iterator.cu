// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/type/arg_index_iterator %S/arg_index_iterator.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --match-full-lines --input-file %T/type/arg_index_iterator/arg_index_iterator.dp.cpp %s
// RUN: %if build_lit %{icpx -c -fsycl -DBUILD_TEST  %T/type/arg_index_iterator/arg_index_iterator.dp.cpp -o %T/type/arg_index_iterator/arg_index_iterator.dp.o %}

#ifndef BUILD_TEST
// CHECK: #include <dpct/dpl_utils.hpp>
#include <cub/cub.cuh>
#include <iostream>

#define N 10
template <typename T>
using Tuple = typename cub::ArgIndexInputIterator<T *>::value_type;

void test1(void) {
  int *d_in;
  // CHECK: dpct::arg_index_input_iterator<int *> Iter(d_in); 
  cub::ArgIndexInputIterator<int *> Iter(d_in);
}

// CHECK: int test2(dpct::arg_index_input_iterator<double *> Iter) {
int test2(cub::ArgIndexInputIterator<double *> Iter) {
  Tuple<double> t = *Iter;
  return t.key;
}

// CHECK: T test3(dpct::arg_index_input_iterator<T *> Iter) {
template <typename T>
T test3(cub::ArgIndexInputIterator<T *> Iter) {
  Tuple<T> t = *Iter;
  return t.value;
}

// CHECK: decltype(auto) test4(dpct::arg_index_input_iterator<IteratorT> Iter) {
template <typename IteratorT>
decltype(auto) test4(cub::ArgIndexInputIterator<IteratorT> Iter) {
  Tuple<typename std::iterator_traits<IteratorT>::value_type> t = *Iter;
  return t.key;
}

class Test5 {
  // CHECK: dpct::arg_index_input_iterator<int *> Iter;
  cub::ArgIndexInputIterator<int *> Iter;

public:
  Test5(int *Ptr) : Iter(Ptr) {}

  // CHECK: dpct::arg_index_input_iterator<int *> getIter() const {
  cub::ArgIndexInputIterator<int *> getIter() const {
    return Iter;
  }
};

// CHECK: using Test6 = dpct::arg_index_input_iterator<int *>; 
using Test6 = cub::ArgIndexInputIterator<int *>;

// CHECK: template <typename T> using Test7 = dpct::arg_index_input_iterator<T *>;
template <typename T> using Test7 = cub::ArgIndexInputIterator<T *>;

// CHECK: template <typename Iterator> using Test8 = dpct::arg_index_input_iterator<Iterator>;
template <typename Iterator> using Test8 = cub::ArgIndexInputIterator<Iterator>;

int test9(void) {
  struct Foo {
    int field;
  };
  // CHECK: auto Ptr = dpct::arg_index_input_iterator<Foo *>(nullptr);
  auto Ptr = cub::ArgIndexInputIterator<Foo *>(nullptr);
  Tuple<Foo> t = *Ptr;
  return t.value.field;
}

void test10(void) {
  int arr[100] = {0};
  // CHECK: dpct::arg_index_input_iterator<int *> Iter(arr);
  cub::ArgIndexInputIterator<int *> Iter(arr);

  // CHECK: Iter = Iter.create_normalize();
  Iter.normalize();
}
#endif
