#include <nvshmem.h>

void test(size_t size) {
  // Start
  nvshmem_malloc(size /*size_t*/);
  // End
}
