// RUN: c2s --format-range=none -out-root %T/try_catch %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/try_catch/try_catch.dp.cpp

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

