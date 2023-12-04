// RUN: cat %s > %T/user_defined_rule.cu
// RUN: cat %S/user_defined_rule.yaml > %T/user_defined_rule.yaml
// RUN: cat %S/user_defined_rule_2.yaml > %T/user_defined_rule_2.yaml
// RUN: cd %T
// RUN: rm -rf %T/user_defined_rule_output
// RUN: mkdir %T/user_defined_rule_output
// RUN: dpct -out-root %T/user_defined_rule_output user_defined_rule.cu --cuda-include-path="%cuda-path/include" --usm-level=none --rule-file=user_defined_rule.yaml --rule-file=user_defined_rule_2.yaml  -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/user_defined_rule_output/user_defined_rule.dp.cpp --match-full-lines user_defined_rule.cu


//CHECK: #ifdef MACRO_A
//CHECK: #include <cmath3>
//CHECK: #include "cmath2"
//CHECK: #endif
//CHECK: #include "aaa.h"
//CHECK: #include "bbb.h"
//CHECK: #include <vector>
//CHECK: #include "ccc.h"
//CHECK: #include "ddd.h"
//CHECK: #include "fruit.h"
#include<iostream>
#include<cmath>
#include<functional>

#define CALL(x) x

void foo3(std::function<int(int)> f){}
int my_min(int a, int b)
{
    return a < b ? a : b;
}



#define VECTOR int
//CHECK: inline void foo() {
__forceinline__ __global__ void foo(){
  int * ptr;
  //CHECK: std::vector<int> a;
  VECTOR a;
  //CHECK: size_t *aaa = foo(ptr, (int *)&(&ptr), dpct::get_out_of_order_queue(),
  //CHECK-NEXT:                   dpct::get_default_context(), dpct::get_current_device());
  cudaMalloc(&ptr, 50);
}

class ClassA{
public:
    int fieldA;
    int fieldC;
    int methodA(int i, int j){return 0;};
};
class ClassB{
public:
  int fieldB;
  int methodB(int i){return 0;};
};

enum Fruit{
  apple,
  banana
};

void foo2(){
  int c = 10;
  int d = 1;
  //CHECK: goo([&](int x) -> int {
  //CHECK-NEXT:   int y = std::min(c, d);
  //CHECK-NEXT:   return std::min(c, d);
  //CHECK-NEXT: });
  foo3([&](int x)->int {
      int y = my_min(c, d);
      return my_min(c, d);
  });
  //CHECK: CALL2(0);
  CALL(0);
  //CHECK: mytype *cu_st;
  CUstream_st *cu_st;

  //CHECK: ClassB a;
  //CHECK-NEXT: a.fieldD = 3;
  //CHECK-NEXT: a.methodB(2);
  //CHECK-NEXT: a.set_a(3);
  //CHECK-NEXT: int k = a.get_a();
  //CHECK-NEXT: Fruit f = pineapple;
  ClassA a;
  a.fieldC = 3;
  a.methodA(1,2);
  a.fieldA = 3;
  int k = a.fieldA;
  Fruit f = Fruit::apple;

  // CHECK: goo([=](int v) {
  // CHECK-NEXT:   int a = std::min(v, 10);
  // CHECK-NEXT:   int b = std::min(v, 100), c = std::min(std::max(v, 10), 100);
  // CHECK-NEXT:   if (v <= 0);
  // CHECK-NEXT:   if (v > 0) return std::min(v, 10);
  // CHECK-NEXT:   if (std::min(std::max(v, 10), 100)) {
  // CHECK-NEXT:     return std::min(std::max(v, 10), 100);
  // CHECK-NEXT:   } else if (std::min(std::max(v, 10), 100)) {
  // CHECK-NEXT:     return std::min(std::max(v, 10), 100);
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     return std::min(std::max(v, 10), 100);
  // CHECK-NEXT:   }
  // CHECK-NEXT: });
  foo3([=](int v){
    int a = ::min(v, 10);
    int b = ::min(v, 100), c = ::min(::max(v, 10), 100);
    if(v <= 0);
    if(v > 0) return ::min(v, 10);
    if(::min(::max(v, 10), 100)){
      return ::min(::max(v, 10), 100);
    } else if(::min(::max(v, 10), 100)){
      return ::min(::max(v, 10), 100);
    } else {
      return ::min(::max(v, 10), 100);
    }});
}

struct MyStruct {
  operator void *() { return NULL; }
};

namespace A1 {
  namespace B1 {
    MyStruct A1B1() { return MyStruct(); }
  }
}

namespace A2 {
  namespace B2 = A1::B1;
}

template<class T> void explicit_template(T a) {}
template<class T> void explicit_template_default(T a) {}

void foo4(){
  // CHECK: A2B2();
  A2::B2::A1B1();
  // CHECK: A2B2();
  ::A2::B2::A1B1();

  int a;
  // CHECK: ET(a);
  explicit_template<int>(a);
  // CHECK: ET(a);
  explicit_template(a);
  
  // CHECK: explicit_template_default<int>(a);
  explicit_template_default<int>(a);
  // CHECK: ET(a);
  explicit_template_default(a);
}
template<typename T>struct OldType{};
// CHECK: void foo5() { NewType<int> *cu_st; }
__device__ void foo5(){ OldType<int> *cu_st;}


__global__ void foo6(){
//CHECK: std::ceil(2.5);
  std::ceil(2.5);
}

int my_max(int a, int b) {
  return 0;
}

void foo7() {
  int a, b;
  //CHECK: my_min(b, b);
  my_max(a, b);
}

template<class T1, class T2>
void foo8(int a) {
  T1 a;
  T2 b;
}

void foo9(){
  //CHECK: foo8<float, int>(0);
  foo8<int, float>(0);
}