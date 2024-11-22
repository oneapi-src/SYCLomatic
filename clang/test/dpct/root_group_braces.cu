// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T/root_group_braces %s --cuda-include-path="%cuda-path/include" --use-experimental-features=root-group -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck %s --match-full-lines --input-file %T/root_group_braces/root_group_braces.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/root_group_braces/root_group_braces.dp.cpp -o %T/root_group_braces/root_group_braces.dp.o %}

#include <cuda_runtime.h>

__global__ void kernel1() {}
__global__ void kernel2() {}

int main() {
  int a = 0;
  // CHECK: case 1:
  // CHECK-NEXT: {
  // CHECK-NEXT: auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};
  // CHECK: }
  // CHECK-NEXT break;
  // CHECK: case 2:
  // CHECK-NEXT: {
  // CHECK-NEXT: auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};
  // CHECK: }
  // CHECK-NEXT break;
  switch (a) {
  case 1:
    kernel1<<<1, 1>>>();
    break;
  case 2:
    kernel2<<<1, 1>>>();
    break;
  default:
    break;
  };
  return 0;
}