// RUN: syclct -out-root %T %s -- -std=c++14 -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck %s --match-full-lines --input-file %T/try_catch.sycl.cpp

namespace Test {
enum class AA : int { ONE,
                      TWO,
                      THREE };
}

__global__ void fun() {}

class B {
public:
// CHECK: B() : data_(Test::AA::ONE) {}
  B() : data_(Test::AA::ONE) {}

private:
  Test::AA data_;
};
