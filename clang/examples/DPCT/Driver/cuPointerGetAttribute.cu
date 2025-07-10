// Option: --usm-level=none
#include <cuda.h>
void test() {
  void *base_ptr;
  void *ptr;
  // Start
  cuPointerGetAttribute(base_ptr/*void **/, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR/*CUpointer_attribute*/, (CUdeviceptr)ptr/*CUdeviceptr*/);
  // End
}