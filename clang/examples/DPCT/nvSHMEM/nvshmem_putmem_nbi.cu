#include <nvshmem.h>

__device__ __host__ void test(void *dest, const void *source, size_t nelems,
                              int pe) {
  // Start
  nvshmem_putmem_nbi(dest /*void **/, source /*const void **/,
                     nelems /*size_t*/, pe /*int*/);
  // End
}
