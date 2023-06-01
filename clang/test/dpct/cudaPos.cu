// RUN: dpct --format-range=none -out-root %T/cudaPos %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cudaPos/cudaPos.dp.cpp --match-full-lines %s

#include <cuda_runtime.h>

struct Bar {

  // CHECK: sycl::id<3> one{0, 0, 0}, two{0, 0, 0}, three{0, 0, 0};
  cudaPos one, two, three;

  // CHECK: sycl::id<3> a{0, 0, 0};
  cudaPos a{};

  // CHECK: sycl::id<3> b{1, 0, 0};
  cudaPos b{1};

  // CHECK: sycl::id<3> c{1, 2, 0};
  cudaPos c{1, 2};

  // CHECK: sycl::id<3> d{1, 2, 3};
  cudaPos d{1, 2, 3};

  // CHECK: sycl::id<3> e = (sycl::id<3>{0, 0, 0});
  cudaPos e = (cudaPos());

  // CHECK: sycl::id<3> f = sycl::id<3>({1, 2, 3});
  cudaPos f = cudaPos({1, 2, 3});

  // CHECK: sycl::id<3> g = sycl::id<3>({1, 2, 0});
  cudaPos g = cudaPos({1, 2});

  // CHECK: Bar(sycl::id<3> a) {}
  Bar(cudaPos a) {}

  // CHECK: Bar(size_t i) : a({i, 0, 0}), b{i, 0, 0}, c{0, 0, 0} {}
  Bar(size_t i) : a({i}), b{i}, c() {}

  Bar(size_t i, size_t j);

  // CHECK: Bar(size_t i, size_t j, size_t k) : a({i, j, k}), b{i, j, k}, c{0, 0, 0} {}
  Bar(size_t i, size_t j, size_t k) : a({i, j, k}), b{i, j, k}, c() {}
};

// CHECK: Bar::Bar(size_t i, size_t j) : a({i, j, 0}), b{i, j, 0}, c{0, 0, 0} {}
Bar::Bar(size_t i, size_t j) : a({i, j}), b{i, j}, c() {}

template <typename T> class A {};

// CHECK: template <typename T = sycl::id<3>> class B {};
template <typename T = cudaPos> class B {};
template <typename... Ts> class C {};


int main() {
  // CHECK: sycl::id<3> a{0, 0, 0}, i{0, 0, 0}, j{0, 0, 0};
  cudaPos a, i, j;

  // CHECK: sycl::id<3> b{1, 0, 0};
  cudaPos b{1};

  // CHECK: sycl::id<3> c{1, 1, 0};
  cudaPos c{1, 1};

  // CHECK: sycl::id<3> d{1, 1, 1};
  cudaPos d{1, 1, 1};

  // CHECK: sycl::id<3> e{0, 0, 0};
  cudaPos e{};

  // CHECK: sycl::id<3> f = sycl::id<3>{0, 0, 0};
  cudaPos f = cudaPos();

  // CHECK: sycl::id<3> g = sycl::id<3>{1, 2, 3};
  cudaPos g = cudaPos{1, 2, 3};

  // CHECK: sycl::id<3> h = (sycl::id<3>)sycl::id<3>{0, 0, 0};
  cudaPos h = (cudaPos)cudaPos();

  // CHECK: sycl::id<3> k = sycl::id<3>(f);
  cudaPos k = cudaPos(f);

  // CHECK: sycl::id<3> m((sycl::id<3>{0, 0, 0}));
  cudaPos m((cudaPos()));

  // CHECK: A<sycl::id<3>> n;
  A<cudaPos> n;

  // CHECK: B<> o;
  B<> o;

  // CHECK: C<sycl::id<3>, sycl::id<3>> p;
  C<cudaPos, cudaPos> p;

  // CHECK: sycl::id<3> *p1;
  cudaPos *p1;

  // CHECK: sycl::id<3> **p2;
  cudaPos **p2;

  // CHECK: sycl::id<3> &r1 = a;
  cudaPos &r1 = a;

  // CHECK: sycl::id<3> &&rval = sycl::id<3>{0, 0, 0};
  cudaPos &&rval = cudaPos();
  return 0;
}
