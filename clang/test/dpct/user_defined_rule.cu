// RUN: cat %s > %T/user_defined_rule.cu
// RUN: cat %S/user_defined_rule.yaml > %T/user_defined_rule.yaml
// RUN: cat %S/user_defined_rule_2.yaml > %T/user_defined_rule_2.yaml
// RUN: cd %T
// RUN: rm -rf %T/user_defined_rule_output
// RUN: mkdir %T/user_defined_rule_output
// RUN: dpct -out-root %T/user_defined_rule_output user_defined_rule.cu --cuda-include-path="%cuda-path/include" --usm-level=none --rule-file=user_defined_rule.yaml --rule-file=user_defined_rule_2.yaml  -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/user_defined_rule_output/user_defined_rule.dp.cpp --match-full-lines user_defined_rule.cu

//CHECK: #include <cmath3>
//CHECK: #include "cmath2"
//CHECK: #include "aaa.h"
//CHECK: #include "bbb.h"
//CHECK: #include <vector>
//CHECK: #include "ccc.h"
#include<iostream>
#include<functional>

#define CALL(x) x

void foo3(std::function<int(int)> f){}
int my_min(int a, int b)
{
    return a < b ? a : b;
}

#include<cmath>

#define VECTOR int
//CHECK: inline void foo() {
__forceinline__ __global__ void foo(){
  int * ptr;
  //CHECK: std::vector<int> a;
  VECTOR a;
  //CHECK: size_t *aaa = foo(ptr, (int *)&(&ptr), dpct::get_default_queue(),
  //CHECK-NEXT:                   dpct::get_default_context(), dpct::get_current_device());
  cudaMalloc(&ptr, 50);
}

void foo2(){
  int c = 10;
  int d = 1;
  //CHECK: goo([&](int x) -> int {
  //CHECK-NEXT:   return std::min(c, d);
  //CHECK-NEXT: });
  foo3([&](int x)->int {
      return my_min(c, d);
  });
  //CHECK: CALL2(0);
  CALL(0);
}