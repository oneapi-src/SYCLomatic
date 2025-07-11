// Option: --use-experimental-features=virtual_mem
#include <cuda.h>
void test() {
  CUmemAllocationProp prop = {};
  CUmemGenericAllocationHandle allocHandle;
  // Start
  cuMemGetAllocationPropertiesFromHandle(&prop/*CUmemAllocationProp **/, allocHandle/*CUmemGenericAllocationHandle*/);
  // End
}