#include <nvshmem.h>

void test(void *ptr) {
  // Start
  nvshmem_free(ptr /*void **/);
  // End
}
