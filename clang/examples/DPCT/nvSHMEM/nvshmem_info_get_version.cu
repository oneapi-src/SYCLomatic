#include <nvshmem.h>

void test(int *major, int *minor) {
  // Start
  nvshmem_info_get_version(major /*int **/, minor /*int **/);
  // End
}
