#include <nvshmemx.h>

__host__ void test() {
  // Start
  nvshmemx_init_attr_t attributes;
  /* 1 */ nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM,
                             &attributes /*nvshmemx_init_attr_t **/);
  /* 2 */ nvshmemx_init_attr(NVSHMEMX_INIT_WITH_SHMEM,
                             &attributes /*nvshmemx_init_attr_t **/);
  // End
}
