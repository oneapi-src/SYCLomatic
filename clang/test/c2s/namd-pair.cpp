// RUN: c2s --format-range=none -out-root %T/namd-pair %s --cuda-include-path="%cuda-path/include" -- -std=c++14
// RUN: FileCheck --input-file %T/namd-pair/namd-pair.cpp.dp.cpp --match-full-lines %s


#include<cuda_runtime.h>
//CHECK: template<bool _Cond, typename _Iftrue, typename _Iffalse>
//CHECK_NEXT: struct conditional {
//CHECK_NEXT:   typedef _Iftrue type;
//CHECK_NEXT: };
//CHECK_NEXT: template<typename _T1>
//CHECK_NEXT: struct pair {
//CHECK_NEXT:   sycl::int2 a;
//CHECK_NEXT:   pair& operator=(typename conditional<true, const pair&, _T1>::type __p) {
//CHECK_NEXT:     return *this;
//CHECK_NEXT:   }
//CHECK_NEXT: };
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
