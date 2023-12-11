// RUN: dpct --format-range=none -out-root %T/cudaExtent %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cudaExtent/cudaExtent.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/cudaExtent/cudaExtent.dp.cpp -o %T/cudaExtent/cudaExtent.dp.o %}

#include <cuda_runtime.h>

struct Bar {

  // CHECK: sycl::range<3> one{0, 0, 0}, two{0, 0, 0}, three{0, 0, 0};
  cudaExtent one, two, three;

  // CHECK: sycl::range<3> a{0, 0, 0};
  cudaExtent a{};

  // CHECK: sycl::range<3> b{1, 0, 0};
  cudaExtent b{1};

  // CHECK: sycl::range<3> c{1, 2, 0};
  cudaExtent c{1, 2};

  // CHECK: sycl::range<3> d{1, 2, 3};
  cudaExtent d{1, 2, 3};

  // CHECK: sycl::range<3> e = (sycl::range<3>{0, 0, 0});
  cudaExtent e = (cudaExtent());

  // CHECK: sycl::range<3> f = sycl::range<3>({1, 2, 3});
  cudaExtent f = cudaExtent({1, 2, 3});

  // CHECK: sycl::range<3> g = sycl::range<3>({1, 2, 0});
  cudaExtent g = cudaExtent({1, 2});

  // CHECK: Bar(sycl::range<3> a) {}
  Bar(cudaExtent a) {}

  // CHECK: Bar(size_t i) : a({i, 0, 0}), b{i, 0, 0}, c{0, 0, 0} {}
  Bar(size_t i) : a({i}), b{i}, c() {}

  Bar(size_t i, size_t j);

  // CHECK: Bar(size_t i, size_t j, size_t k) : a({i, j, k}), b{i, j, k}, c{0, 0, 0} {}
  Bar(size_t i, size_t j, size_t k) : a({i, j, k}), b{i, j, k}, c() {}
};

// CHECK: Bar::Bar(size_t i, size_t j) : a({i, j, 0}), b{i, j, 0}, c{0, 0, 0} {}
Bar::Bar(size_t i, size_t j) : a({i, j}), b{i, j}, c() {}

template <typename T> class A {};

// CHECK: template <typename T = sycl::range<3>> class B {};
template <typename T = cudaExtent> class B {};
template <typename... Ts> class C {};


int main() {
  // CHECK: sycl::range<3> a{0, 0, 0}, i{0, 0, 0}, j{0, 0, 0};
  cudaExtent a, i, j;

  // CHECK: sycl::range<3> b{1, 0, 0};
  cudaExtent b{1};

  // CHECK: sycl::range<3> c{1, 1, 0};
  cudaExtent c{1, 1};

  // CHECK: sycl::range<3> d{1, 1, 1};
  cudaExtent d{1, 1, 1};

  // CHECK: sycl::range<3> e{0, 0, 0};
  cudaExtent e{};

  // CHECK: sycl::range<3> f = sycl::range<3>{0, 0, 0};
  cudaExtent f = cudaExtent();

  // CHECK: sycl::range<3> g = sycl::range<3>{1, 2, 3};
  cudaExtent g = cudaExtent{1, 2, 3};

  // CHECK: sycl::range<3> h = (sycl::range<3>)sycl::range<3>{0, 0, 0};
  cudaExtent h = (cudaExtent)cudaExtent();

  // CHECK: sycl::range<3> k = sycl::range<3>(f);
  cudaExtent k = cudaExtent(f);

  // CHECK: sycl::range<3> l = sycl::range<3>({1, 2, 3});
  cudaExtent l = cudaExtent({1, 2, 3});

  // CHECK: sycl::range<3> m((sycl::range<3>{0, 0, 0}));
  cudaExtent m((cudaExtent()));

  // CHECK: A<sycl::range<3>> n;
  A<cudaExtent> n;

  // CHECK: B<> o;
  B<> o;

  // CHECK: C<sycl::range<3>, sycl::range<3>> p;
  C<cudaExtent, cudaExtent> p;

  // CHECK: sycl::range<3> *p1;
  cudaExtent *p1;

  // CHECK: sycl::range<3> **p2;
  cudaExtent **p2;

  // CHECK: sycl::range<3> &r1 = a;
  cudaExtent &r1 = a;

  // CHECK: sycl::range<3> &&rval = sycl::range<3>{0, 0, 0};
  cudaExtent &&rval = cudaExtent();
  return 0;
}
