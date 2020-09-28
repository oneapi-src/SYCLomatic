// RUN: dpct --format-range=none --out-root %T %s %S/atomic_no_warning2.cu --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/atomic_no_warning.dp.hpp --match-full-lines %S/atomic_no_warning.cuh

// CHECK: #include <CL/sycl.hpp>
#include <cuda_runtime.h>
#include "atomic_no_warning.cuh"

__device__ void foo(){
  int a, b;
  ATOMIC_ADD( a, b )
}