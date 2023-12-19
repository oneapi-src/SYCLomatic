// RUN: dpct --format-range=none -out-root %T/insert_sycl_header_onlyonce %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/insert_sycl_header_onlyonce/insert_sycl_header_onlyonce.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/insert_sycl_header_onlyonce/insert_sycl_header_onlyonce.dp.cpp -o %T/insert_sycl_header_onlyonce/insert_sycl_header_onlyonce.dp.o %}

// CHECK: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK: #include <stdio.h>
// CHECK-NOT:#include <sycl/sycl.hpp>
#include <stdio.h>
#include <cuda.h>

int main(){
  return 0;
}


