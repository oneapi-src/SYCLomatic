#include <nvshmemx.h>

__device__ __host__ void test(uint64_t *sig_addr, uint64_t signal, int pe) {
  // Start
  /* 1 */ nvshmemx_signal_op(sig_addr /*uint64_t **/, signal /*uint64_t*/,
                             NVSHMEM_SIGNAL_SET, pe /*int*/);
  /* 2 */ nvshmemx_signal_op(sig_addr /*uint64_t **/, signal /*uint64_t*/,
                             NVSHMEM_SIGNAL_ADD, pe /*int*/);
  // End
}
