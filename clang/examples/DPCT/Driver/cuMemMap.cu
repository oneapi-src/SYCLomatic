// Option: --use-experimental-features=virtual_mem
#include <cuda.h>
void test(CUdeviceptr ptr, size_t size, size_t offset,
          unsigned long long flags) {
  // Start
  CUmemGenericAllocationHandle handle;
  cuMemMap(ptr /*CUdeviceptr*/, size /*size_t*/, offset /*size_t*/,
           handle /*CUmemGenericAllocationHandle*/,
           flags /*unsigned long long */);
  // End
}