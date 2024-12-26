// Option: --use-experimental-features=virtual_mem
#include <cuda.h>
void test(CUdeviceptr *ptr, size_t size, size_t alignment, CUdeviceptr addr,
          unsigned long long flags) {
  // Start
  cuMemAddressReserve(ptr /*CUdeviceptr **/, size /*size_t*/,
                      alignment /*size_t*/, addr /*CUdeviceptr*/,
                      flags /*unsigned long long*/);
  // End
}