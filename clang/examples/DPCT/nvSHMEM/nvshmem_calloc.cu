#include <nvshmem.h>

void test(size_t count, size_t size) {
  // Start
  nvshmem_calloc(count /*size_t*/, size /*size_t*/);
  // End
}
