// RUN: syclct -out-root %T %s -passes "ErrorTryCatchRule" -- -std=c++11 -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck %s --match-full-lines --input-file %T/error-handling-trycatch.sycl.cpp

// CHECK:void test_simple() try {
// CHECK-NEXT:}
// CHECK-NEXT:catch (cl::sycl::exception const &exc) {
// CHECK-NEXT:  std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK-NEXT:  std::exit(1);
// CHECK-NEXT:}
void test_simple() {
}
// CHECK:void test_simple_1()
// CHECK-NEXT:try {
// CHECK-NEXT:{{ +}}
// CHECK-NEXT:}
// CHECK-NEXT:catch (cl::sycl::exception const &exc) {
// CHECK-NEXT:  std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK-NEXT:  std::exit(1);
// CHECK-NEXT:}
void test_simple_1()
{
  
}
// CHECK:void test_simple_2() try {}
// CHECK-NEXT:catch (cl::sycl::exception const &exc) {
// CHECK-NEXT:  std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK-NEXT:  std::exit(1);
// CHECK-NEXT:}
void test_simple_2() {}

// CHECK:void test_lambda() try {
// CHECK-NEXT:  auto f = []() {
// CHECK-NEXT:  };
// CHECK-NEXT:}
// CHECK-NEXT:catch (cl::sycl::exception const &exc) {
// CHECK-NEXT:  std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK-NEXT:  std::exit(1);
// CHECK-NEXT:}
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

// CHECK:struct A : public Base {
// CHECK-NEXT:  int a;
// CHECK-NEXT:  A(int a)try : a(a) {}
// CHECK-NEXT:catch (cl::sycl::exception const &exc) {
// CHECK-NEXT:  std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK-NEXT:  std::exit(1);
// CHECK-NEXT:}
// CHECK-NEXT:  ~A() throw() try {}
// CHECK-NEXT:catch (cl::sycl::exception const &exc) {
// CHECK-NEXT:  std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK-NEXT:  std::exit(1);
// CHECK-NEXT:}
// CHECK-NEXT:};

struct A : public Base {
  int a;
  A(int a): a(a) {}
  ~A() throw() {}
};

// CHECK:struct B {
// CHECK-NEXT:  B() try {}
// CHECK-NEXT:catch (cl::sycl::exception const &exc) {
// CHECK-NEXT:  std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK-NEXT:  std::exit(1);
// CHECK-NEXT:}
// CHECK-NEXT:  ~B() throw() try {}
// CHECK-NEXT:catch (cl::sycl::exception const &exc) {
// CHECK-NEXT:  std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK-NEXT:  std::exit(1);
// CHECK-NEXT:}
// CHECK-NEXT:};
struct B {
  B() {}
  ~B() throw() {}
};