// RUN: cu2sycl -out-root %T %s -passes "ErrorTryCatchRule" -- -std=c++11 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/error-handling-trycatch.sycl.cpp

// CHECK:void test_simple() try {
// CHECK-NEXT:}
// CHECK-NEXT:catch (cl::sycl::exception const &exc) {
// CHECK-NEXT:  std::cerr << exc.what() << "EOE" << std::endl;
// CHECK-NEXT:  std::exit(1);
// CHECK-NEXT:}
void test_simple() {
}
// CHECK:void test_simple_1()
// CHECK-NEXT:try {
// CHECK-NEXT:{{ +}}
// CHECK-NEXT:}
// CHECK-NEXT:catch (cl::sycl::exception const &exc) {
// CHECK-NEXT:  std::cerr << exc.what() << "EOE" << std::endl;
// CHECK-NEXT:  std::exit(1);
// CHECK-NEXT:}
void test_simple_1()
{
  
}
// CHECK:void test_simple_2() try {}
// CHECK-NEXT:catch (cl::sycl::exception const &exc) {
// CHECK-NEXT:  std::cerr << exc.what() << "EOE" << std::endl;
// CHECK-NEXT:  std::exit(1);
// CHECK-NEXT:}
void test_simple_2() {}

// CHECK:void test_lambda() try {
// CHECK-NEXT:  auto f = []() {
// CHECK-NEXT:  };
// CHECK-NEXT:}
// CHECK-NEXT:catch (cl::sycl::exception const &exc) {
// CHECK-NEXT:  std::cerr << exc.what() << "EOE" << std::endl;
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