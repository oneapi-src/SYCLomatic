// Option: --use-experimental-features=virtual_mem
#include <cuda.h>
void test() {
  // Start
  CUmemGenericAllocationHandle handle;
  cuMemRelease(handle /*CUmemGenericAllocationHandle */);
  // End
}