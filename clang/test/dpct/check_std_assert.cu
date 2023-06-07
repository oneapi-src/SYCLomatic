// RUN: dpct --format-range=none -out-root %T/check_std_assert %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/check_std_assert/check_std_assert.dp.cpp

#include <cassert>
#include <thrust/copy.h>

int main(){
  // CHECK: int a=1;
  // CHECK-NEXT: // The function '__assert_fail' with the same name as cuda runtime will be used in assert func
  // CHECK-NEXT: // Non-CUDA APIs should not be touched or emit warning msgs
  // CHECK-NEXT: assert(a==0);
  // CHECK-NEXT: return 1;
  int a=1;
  // The function '__assert_fail' with the same name as cuda runtime will be used in assert func
  // Non-CUDA APIs should not be touched or emit warning msgs
  assert(a==0);
  return 1;
}