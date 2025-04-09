// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2
// RUN: dpct --format-range=none --usm-level=none -out-root %T/buf %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++11
// RUN: FileCheck --match-full-lines --input-file %T/buf/cudaGetPointer_none_usm.dp.cpp %s
// RUN: %if build_lit %{icpx -c -fsycl %T/buf/cudaGetPointer_none_usm.dp.cpp -o %T/buf/cudaGetPointer_none_usm.dp.o %}

#include <cuda.h>
void test_attribute() {
  void *base_ptr;
  void *ptr;
  //CHECK:  if (DPCT_CHECK_ERROR(base_ptr = dpct::get_base_addr((dpct::device_ptr)ptr)) != 0);
  if (cuPointerGetAttribute(base_ptr, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, (CUdeviceptr)ptr) != CUDA_SUCCESS);
}
