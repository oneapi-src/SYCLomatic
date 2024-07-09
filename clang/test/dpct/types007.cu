// UNSUPPORTED: cuda-8.0, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/types007 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17 -fsized-deallocation
// RUN: FileCheck %s --match-full-lines --input-file %T/types007/types007.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/types007/types007.dp.cpp -o %T/types007/types007.dp.o %}

#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/optional.h>
#include <thrust/pair.h>

#include "cublas_v2.h"

int main(int argc, char **argv) {
  // CHECK:  std::optional<int> b = std::nullopt;
  // CHECK-NEXT:  std::optional<int> c = 1;
  thrust::optional<int> b = thrust::nullopt;
  thrust::optional<int> c = 1;
}

// CHECK: void foo_1(dpct::compute_type a1) {
// CHECK-NEXT:   dpct::compute_type b1 = a1;
// CHECK-NEXT: }
void foo_1(cublasComputeType_t a1) {
  cublasComputeType_t b1 = a1;
}
