// REQUIRES: system-linux
// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.2

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=nvshmem_putmem_signal_nbi | FileCheck %s -check-prefix=NVSHMEM_PUTMEM_SIGNAL_NBI
// NVSHMEM_PUTMEM_SIGNAL_NBI: CUDA API:
// NVSHMEM_PUTMEM_SIGNAL_NBI-NEXT:   /* 1 */ nvshmem_putmem_signal_nbi(dest /*void **/, source /*const void **/,
// NVSHMEM_PUTMEM_SIGNAL_NBI-NEXT:                                     nelems /*size_t*/, sig_addr /*uint64_t **/,
// NVSHMEM_PUTMEM_SIGNAL_NBI-NEXT:                                     signal /*uint64_t*/, NVSHMEM_SIGNAL_SET,
// NVSHMEM_PUTMEM_SIGNAL_NBI-NEXT:                                     pe /*int*/);
// NVSHMEM_PUTMEM_SIGNAL_NBI-NEXT:   /* 2 */ nvshmem_putmem_signal_nbi(dest /*void **/, source /*const void **/,
// NVSHMEM_PUTMEM_SIGNAL_NBI-NEXT:                                     nelems /*size_t*/, sig_addr /*uint64_t **/,
// NVSHMEM_PUTMEM_SIGNAL_NBI-NEXT:                                     signal /*uint64_t*/, NVSHMEM_SIGNAL_ADD,
// NVSHMEM_PUTMEM_SIGNAL_NBI-NEXT:                                     pe /*int*/);
// NVSHMEM_PUTMEM_SIGNAL_NBI-NEXT: Is migrated to:
// NVSHMEM_PUTMEM_SIGNAL_NBI-NEXT:   /* 1 */ ishmem_putmem_signal_nbi(dest, source, nelems, sig_addr, signal, ISHMEM_SIGNAL_SET, pe);
// NVSHMEM_PUTMEM_SIGNAL_NBI-NEXT:   /* 2 */ ishmem_putmem_signal_nbi(dest, source, nelems, sig_addr, signal, ISHMEM_SIGNAL_ADD, pe);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=nvshmem_signal_wait_until | FileCheck %s -check-prefix=NVSHMEM_SIGNAL_WAIT_UNTIL
// NVSHMEM_SIGNAL_WAIT_UNTIL: CUDA API:
// NVSHMEM_SIGNAL_WAIT_UNTIL-NEXT:   /* 1 */ nvshmem_signal_wait_until(sig_addr /*uint64_t **/, NVSHMEM_CMP_EQ,
// NVSHMEM_SIGNAL_WAIT_UNTIL-NEXT:                                     cmp_val /*uint64_t*/);
// NVSHMEM_SIGNAL_WAIT_UNTIL-NEXT:   /* 2 */ nvshmem_signal_wait_until(sig_addr /*uint64_t **/, NVSHMEM_CMP_NE,
// NVSHMEM_SIGNAL_WAIT_UNTIL-NEXT:                                     cmp_val /*uint64_t*/);
// NVSHMEM_SIGNAL_WAIT_UNTIL-NEXT:   /* 3 */ nvshmem_signal_wait_until(sig_addr /*uint64_t **/, NVSHMEM_CMP_GT,
// NVSHMEM_SIGNAL_WAIT_UNTIL-NEXT:                                     cmp_val /*uint64_t*/);
// NVSHMEM_SIGNAL_WAIT_UNTIL-NEXT:   /* 4 */ nvshmem_signal_wait_until(sig_addr /*uint64_t **/, NVSHMEM_CMP_GE,
// NVSHMEM_SIGNAL_WAIT_UNTIL-NEXT:                                     cmp_val /*uint64_t*/);
// NVSHMEM_SIGNAL_WAIT_UNTIL-NEXT:   /* 5 */ nvshmem_signal_wait_until(sig_addr /*uint64_t **/, NVSHMEM_CMP_LT,
// NVSHMEM_SIGNAL_WAIT_UNTIL-NEXT:                                     cmp_val /*uint64_t*/);
// NVSHMEM_SIGNAL_WAIT_UNTIL-NEXT:   /* 6 */ nvshmem_signal_wait_until(sig_addr /*uint64_t **/, NVSHMEM_CMP_LE,
// NVSHMEM_SIGNAL_WAIT_UNTIL-NEXT:                                     cmp_val /*uint64_t*/);
// NVSHMEM_SIGNAL_WAIT_UNTIL-NEXT: Is migrated to:
// NVSHMEM_SIGNAL_WAIT_UNTIL-NEXT:   /* 1 */ ishmem_signal_wait_until(sig_addr, ISHMEM_CMP_EQ, cmp_val);
// NVSHMEM_SIGNAL_WAIT_UNTIL-NEXT:   /* 2 */ ishmem_signal_wait_until(sig_addr, ISHMEM_CMP_NE, cmp_val);
// NVSHMEM_SIGNAL_WAIT_UNTIL-NEXT:   /* 3 */ ishmem_signal_wait_until(sig_addr, ISHMEM_CMP_GT, cmp_val);
// NVSHMEM_SIGNAL_WAIT_UNTIL-NEXT:   /* 4 */ ishmem_signal_wait_until(sig_addr, ISHMEM_CMP_GE, cmp_val);
// NVSHMEM_SIGNAL_WAIT_UNTIL-NEXT:   /* 5 */ ishmem_signal_wait_until(sig_addr, ISHMEM_CMP_LT, cmp_val);
// NVSHMEM_SIGNAL_WAIT_UNTIL-NEXT:   /* 6 */ ishmem_signal_wait_until(sig_addr, ISHMEM_CMP_LE, cmp_val);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=nvshmemx_signal_op | FileCheck %s -check-prefix=NVSHMEMX_SIGNAL_OP
// NVSHMEMX_SIGNAL_OP: CUDA API:
// NVSHMEMX_SIGNAL_OP-NEXT:   /* 1 */ nvshmemx_signal_op(sig_addr /*uint64_t **/, signal /*uint64_t*/,
// NVSHMEMX_SIGNAL_OP-NEXT:                              NVSHMEM_SIGNAL_SET, pe /*int*/);
// NVSHMEMX_SIGNAL_OP-NEXT:   /* 2 */ nvshmemx_signal_op(sig_addr /*uint64_t **/, signal /*uint64_t*/,
// NVSHMEMX_SIGNAL_OP-NEXT:                              NVSHMEM_SIGNAL_ADD, pe /*int*/);
// NVSHMEMX_SIGNAL_OP-NEXT: Is migrated to:
// NVSHMEMX_SIGNAL_OP-NEXT:   /* 1 */ dpct::shmemx::signal_op(sig_addr, signal, ISHMEM_SIGNAL_SET, pe);
// NVSHMEMX_SIGNAL_OP-NEXT:   /* 2 */ dpct::shmemx::signal_op(sig_addr, signal, ISHMEM_SIGNAL_ADD, pe);
