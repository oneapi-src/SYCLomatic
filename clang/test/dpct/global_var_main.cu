// RUN: dpct --format-range=none -out-root %T/global_var_main %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/global_var_main/global_var_main.dp.cpp --match-full-lines %s
// RUN: FileCheck --input-file %T/global_var_main/global_var_definition.dp.hpp --match-full-lines %S/global_var_definition.cuh

#include <cuda.h>
#include <stdio.h>

#include "global_var_definition.cuh"

__global__ void kernel(){
  int a = c_clusters[0];
}

//CHECK:extern "C" int foo(){
//CHECK-NEXT:  dpct::get_default_queue().submit(
//CHECK-NEXT:    [&](sycl::handler &cgh) {
//CHECK-NEXT:      c_clusters.init();
//CHECK-EMPTY:
//CHECK-NEXT:      auto c_clusters_ptr_ct1 = c_clusters.get_ptr();
//CHECK-EMPTY:
//CHECK-NEXT:      cgh.parallel_for(
//CHECK-NEXT:        sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
//CHECK-NEXT:        [=](sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:          kernel(c_clusters_ptr_ct1);
//CHECK-NEXT:        });
//CHECK-NEXT:    });
//CHECK-NEXT:  return 0;
//CHECK-NEXT:}
extern "C" int foo(){
  kernel<<<1,1,1>>>();
  return 0;
}

int main(){
  return 0;
}
