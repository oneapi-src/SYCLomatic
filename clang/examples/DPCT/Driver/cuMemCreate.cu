// Option: --use-experimental-features=virtual_mem
#include <cuda.h>
void test(size_t size, CUmemAllocationProp *prop, unsigned long long flags) {
  // Start
  CUmemGenericAllocationHandle *handle;
  cuMemCreate(handle /*CUmemGenericAllocationHandle **/, size /*size_t*/,
              prop /*CUmemAllocationProp **/, flags /*unsigned long long*/);
  // End
}