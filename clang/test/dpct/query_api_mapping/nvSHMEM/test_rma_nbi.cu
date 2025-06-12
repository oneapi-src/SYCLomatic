// REQUIRES: system-linux
// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.2

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=nvshmem_putmem_nbi | FileCheck %s -check-prefix=NVSHMEM_PUTMEM_NBI
// NVSHMEM_PUTMEM_NBI: CUDA API:
// NVSHMEM_PUTMEM_NBI-NEXT:   nvshmem_putmem_nbi(dest /*void **/, source /*const void **/,
// NVSHMEM_PUTMEM_NBI-NEXT:                      nelems /*size_t*/, pe /*int*/);
// NVSHMEM_PUTMEM_NBI-NEXT: Is migrated to:
// NVSHMEM_PUTMEM_NBI-NEXT:   ishmem_putmem_nbi(dest, source, nelems, pe);
