// RUN: syclct -out-root %T %s  -- -std=c++14 -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck %s --match-full-lines --input-file %T/error-handling-trycatch.sycl.cpp

void test_simple() {
}
void test_simple_1()
{

}
void test_simple_2() {}

void test_lambda() {
  auto f = []() {
  };
}
// CHECK:extern void test_notrycatch(int);
extern void test_notrycatch(int);

// CHECK: void test_00() {}
__global__
void test_00() {}

struct Base {
  int aa = 0;
};

struct A : public Base {
  int a;
  A(int a): a(a) {}
  ~A() throw() {}
};

struct B {
  B() {}
  ~B() throw() {}
};

struct C : public Base {
  C() {}
  ~C() throw() {}
};
