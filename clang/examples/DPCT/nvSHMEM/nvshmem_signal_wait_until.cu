#include <nvshmem.h>

__device__ __host__ void test(uint64_t *sig_addr, uint64_t cmp_val) {
  // Start
  /* 1 */ nvshmem_signal_wait_until(sig_addr /*uint64_t **/, NVSHMEM_CMP_EQ,
                                    cmp_val /*uint64_t*/);
  /* 2 */ nvshmem_signal_wait_until(sig_addr /*uint64_t **/, NVSHMEM_CMP_NE,
                                    cmp_val /*uint64_t*/);
  /* 3 */ nvshmem_signal_wait_until(sig_addr /*uint64_t **/, NVSHMEM_CMP_GT,
                                    cmp_val /*uint64_t*/);
  /* 4 */ nvshmem_signal_wait_until(sig_addr /*uint64_t **/, NVSHMEM_CMP_GE,
                                    cmp_val /*uint64_t*/);
  /* 5 */ nvshmem_signal_wait_until(sig_addr /*uint64_t **/, NVSHMEM_CMP_LT,
                                    cmp_val /*uint64_t*/);
  /* 6 */ nvshmem_signal_wait_until(sig_addr /*uint64_t **/, NVSHMEM_CMP_LE,
                                    cmp_val /*uint64_t*/);
  // End
}
