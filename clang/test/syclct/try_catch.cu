// RUN: syclct -out-root %T %s -- -std=c++11 -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck %s --match-full-lines --input-file %T/try_catch.sycl.cpp

namespace Test {
enum class AA : int { ONE,
                      TWO,
                      THREE };
}

class B {
public:
// CHECK: B() try : data_(Test::AA::ONE) {}
// CHECK: catch (cl::sycl::exception const &exc) {
// CHECK:   std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
// CHECK:   std::exit(1);
// CHECK: }
  B() : data_(Test::AA::ONE) {}

private:
  Test::AA data_;
};
