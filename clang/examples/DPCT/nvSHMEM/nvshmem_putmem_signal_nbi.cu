#include <nvshmem.h>

__device__ __host__ void test(void *dest, const void *source, size_t nelems,
                              uint64_t *sig_addr, uint64_t signal, int pe) {
  // Start
  /* 1 */ nvshmem_putmem_signal_nbi(dest /*void **/, source /*const void **/,
                                    nelems /*size_t*/, sig_addr /*uint64_t **/,
                                    signal /*uint64_t*/, NVSHMEM_SIGNAL_SET,
                                    pe /*int*/);
  /* 2 */ nvshmem_putmem_signal_nbi(dest /*void **/, source /*const void **/,
                                    nelems /*size_t*/, sig_addr /*uint64_t **/,
                                    signal /*uint64_t*/, NVSHMEM_SIGNAL_ADD,
                                    pe /*int*/);
  // End
}
