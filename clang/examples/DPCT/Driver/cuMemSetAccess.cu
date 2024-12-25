// Option: --use-experimental-features=virtual_mem
#include <cuda.h>
void test(CUdeviceptr ptr, size_t size, CUmemAccessDesc *desc, size_t count) {
  // Start
  cuMemSetAccess(ptr /*CUdeviceptr*/, size /*size_t*/,
                 desc /*CUmemAccessDesc **/, count /*size_t*/);
  // End
}