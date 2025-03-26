// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2
// RUN: dpct --format-range=none --usm-level=restricted -out-root %T/device_prop %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++11
// RUN: FileCheck --match-full-lines --input-file %T/device_prop/cudaGetPointer.dp.cpp %s


#include <cuda.h>
void test_attribute() {
  void *base_ptr;
  void *ptr;
  // CHECK:/*
  // CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'CU_POINTER_ATTRIBUTE_RANGE_START_ADDR' parameter could not be migrated. You may need to update the code manually.
  // CHECK-NEXT:*/
  if (cuPointerGetAttribute(base_ptr, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, (CUdeviceptr)ptr) != CUDA_SUCCESS);
}
