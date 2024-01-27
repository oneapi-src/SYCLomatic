// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct -out-root %T/match_intrinsic %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/match_intrinsic/match_intrinsic.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/match_intrinsic/match_intrinsic.dp.cpp -o %T/match_intrinsic/match_intrinsic.dp.o %}

#include "cuda.h"
#include "cuda_runtime.h"

__global__ void match_any() {
  unsigned mask;
  int val;
  // CHECK: dpct::match_any_over_sub_group(item_ct1.get_sub_group(), mask, val);
  __match_any_sync(mask, val);
}

__global__ void match_any2() {
  unsigned mask;
  int * val;
  // CHECK: dpct::match_any_over_sub_group(item_ct1.get_sub_group(), mask, (unsigned long long)val);
  __match_any_sync(mask, (unsigned long long)val);
}

__global__ void match_all() {
  unsigned mask;
  int val;
  int pred;
  // CHECK: dpct::match_all_over_sub_group(item_ct1.get_sub_group(), mask, val, &pred);
  __match_all_sync(mask, val, &pred);
}

int main() {
  match_any<<<1, 32>>>();
  match_all<<<1, 32>>>();
  return 0;
}
