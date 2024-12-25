// Option: --use-experimental-features=virtual_mem
#include <cuda.h>
void test(size_t *granularity, CUmemAllocationProp *prop,
          CUmemAllocationGranularity_flags option) {
  // Start
  cuMemGetAllocationGranularity(granularity /*size_t
                                             **/
                                ,
                                prop /*CUmemAllocationProp **/,
                                option /*CUmemAllocationGranularity_flags*/);
  // End
}