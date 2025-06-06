#include <nvshmem.h>

void test(size_t alignment, size_t size) {
  // Start
  nvshmem_align(alignment /*size_t*/, size /*size_t*/);
  // End
}
