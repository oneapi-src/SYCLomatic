// Option: --use-experimental-features=virtual_mem
#include <cuda.h>
void test(CUdeviceptr ptr, size_t size) {
  // Start
  cuMemUnmap(ptr /*CUdeviceptr*/, size /*size_t*/);
  // End
}