#include <nvshmem.h>

void test(void *local_ptr, const void *dest, int pe) {
  // Start
  local_ptr = nvshmem_ptr(dest /*const void **/, pe /*int*/);
  // End
}
