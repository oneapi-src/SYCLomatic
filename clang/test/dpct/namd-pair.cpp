// RUN: dpct --format-range=none -out-root %T/namd-pair %s --cuda-include-path="%cuda-path/include" -- -std=c++14
// RUN: FileCheck --input-file %T/namd-pair/namd-pair.cpp.dp.cpp --match-full-lines %s


#include<cuda_runtime.h>
//CHECK: template<bool _Cond, typename _Iftrue, typename _Iffalse>
//CHECK-NEXT: struct conditional {
//CHECK-NEXT:   typedef _Iftrue type;
//CHECK-NEXT: };
//CHECK-NEXT: template<typename _T1>
//CHECK-NEXT: struct pair {
//CHECK-NEXT:   sycl::int2 a;
//CHECK-NEXT:   pair& operator=(typename conditional<true, const pair&, _T1>::type __p) {
//CHECK-NEXT:     return *this;
//CHECK-NEXT:   }
//CHECK-NEXT: };
template<bool _Cond, typename _Iftrue, typename _Iffalse>
struct conditional {
  typedef _Iftrue type;
};
template<typename _T1>
struct pair {
  int2 a;
  pair& operator=(typename conditional<true, const pair&, _T1>::type __p) {
    return *this;
  }
};
