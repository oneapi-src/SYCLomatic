// RUN: dpct --format-range=none --out-root %T/kernel_namespace %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --match-full-lines --input-file %T/kernel_namespace/kernel_namespace.dp.cpp %s

#include "cuda_runtime.h"
#include <iostream>

__global__ void k1(){}

namespace aaa {
__global__ void k2(){}

namespace bbb {
__global__ void k3(){}
} // namespace bbb
} // namespace aaa


void foo1() {
  // CHECK: k1();
  k1<<<1,1>>>();
}

void foo2() {
  // CHECK: aaa::k2();
  aaa::k2<<<1,1>>>();
}

void foo3() {
  // CHECK: aaa::bbb::k3();
  aaa::bbb::k3<<<1,1>>>();
}


namespace aaa {
void foo4() {
  // CHECK: k1();
  k1<<<1,1>>>();
}

void foo5() {
  // CHECK: k2();
  k2<<<1,1>>>();
}

void foo6() {
  // CHECK: bbb::k3();
  bbb::k3<<<1,1>>>();
}

namespace bbb {
void foo7() {
  // CHECK: k1();
  k1<<<1,1>>>();
}

void foo8() {
  // CHECK: k2();
  k2<<<1,1>>>();
}

void foo9() {
  // CHECK: k3();
  k3<<<1,1>>>();
}
} // namespace bbb
} // namespace aaa


namespace ccc{
void foo10() {
  // CHECK: k1();
  k1<<<1,1>>>();
}

void foo11() {
  // CHECK: aaa::k2();
  aaa::k2<<<1,1>>>();
}

void foo12() {
  // CHECK: aaa::bbb::k3();
  aaa::bbb::k3<<<1,1>>>();
}
} // namespace ccc


namespace {
  __global__ void k4(){}
}
void foo13() {
  // CHECK: k4();
  k4<<<1,1>>>();
}

