// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/type/discard_iterator %S/discard_iterator.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/type/discard_iterator/discard_iterator.dp.cpp %s

// CHECK: #include <oneapi/dpl/iterator>
#include <cub/cub.cuh>
#include <iostream>

void test1(void) {
  // CHECK: oneapi::dpl::discard_iterator Iter; 
  cub::DiscardOutputIterator<> Iter;
}

// CHECK: int test2(oneapi::dpl::discard_iterator Iter)
void test2(cub::DiscardOutputIterator<double *> Iter) {
  *Iter = 1.0;
}

// CHECK: T test3(oneapi::dpl::discard_iterator Iter)
template <typename T>
void test3(cub::DiscardOutputIterator<T> Iter) {
  (void) Iter;
}

class Test4 {
  // CHECK: oneapi::dpl::discard_iterator Iter;
  cub::DiscardOutputIterator<> Iter;
public:
  // CHECK: oneapi::dpl::discard_iterator getIter() const
  cub::DiscardOutputIterator<> getIter() const {
    return Iter;
  }
};

class Test5 {
  // CHECK: oneapi::dpl::discard_iterator Iter;
  cub::DiscardOutputIterator<int> Iter;

public:
  // CHECK: oneapi::dpl::discard_iterator getIter() const
  cub::DiscardOutputIterator<int> getIter() const {
    return Iter;
  }
};

// CHECK: using Test6 = oneapi::dpl::discard_iterator; 
using Test6 = cub::DiscardOutputIterator<int>;

// CHECK: template <typename T> using Test7 = oneapi::dpl::discard_iterator;
template <typename T> using Test7 = cub::DiscardOutputIterator<T>;
