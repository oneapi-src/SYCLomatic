// REQUIRES: system-linux
// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=nvshmem_finalize | FileCheck %s -check-prefix=NVSHMEM_FINALIZE
// NVSHMEM_FINALIZE: CUDA API:
// NVSHMEM_FINALIZE-NEXT:   nvshmem_finalize(/*void*/);
// NVSHMEM_FINALIZE-NEXT: Is migrated to:
// NVSHMEM_FINALIZE-NEXT:   ishmem_finalize();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=nvshmem_info_get_name | FileCheck %s -check-prefix=NVSHMEM_INFO_GET_NAME
// NVSHMEM_INFO_GET_NAME: CUDA API:
// NVSHMEM_INFO_GET_NAME-NEXT:   nvshmem_info_get_name(name /*char **/);
// NVSHMEM_INFO_GET_NAME-NEXT: Is migrated to:
// NVSHMEM_INFO_GET_NAME-NEXT:   ishmem_info_get_name(name);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=nvshmem_info_get_version | FileCheck %s -check-prefix=NVSHMEM_INFO_GET_VERSION
// NVSHMEM_INFO_GET_VERSION: CUDA API:
// NVSHMEM_INFO_GET_VERSION-NEXT:   nvshmem_info_get_version(major /*int **/, minor /*int **/);
// NVSHMEM_INFO_GET_VERSION-NEXT: Is migrated to:
// NVSHMEM_INFO_GET_VERSION-NEXT:   ishmem_info_get_version(major, minor);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=nvshmem_init | FileCheck %s -check-prefix=NVSHMEM_INIT
// NVSHMEM_INIT: CUDA API:
// NVSHMEM_INIT-NEXT:   nvshmem_init(/*void*/);
// NVSHMEM_INIT-NEXT: Is migrated to:
// NVSHMEM_INIT-NEXT:   ishmem_init();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=nvshmem_my_pe | FileCheck %s -check-prefix=NVSHMEM_MY_PE
// NVSHMEM_MY_PE: CUDA API:
// NVSHMEM_MY_PE-NEXT:   nvshmem_my_pe(/*void*/);
// NVSHMEM_MY_PE-NEXT: Is migrated to:
// NVSHMEM_MY_PE-NEXT:   ishmem_my_pe();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=nvshmem_n_pes | FileCheck %s -check-prefix=NVSHMEM_N_PES
// NVSHMEM_N_PES: CUDA API:
// NVSHMEM_N_PES-NEXT:   nvshmem_n_pes(/*void*/);
// NVSHMEM_N_PES-NEXT: Is migrated to:
// NVSHMEM_N_PES-NEXT:   ishmem_n_pes();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=nvshmem_ptr | FileCheck %s -check-prefix=NVSHMEM_PTR
// NVSHMEM_PTR: CUDA API:
// NVSHMEM_PTR-NEXT:   local_ptr = nvshmem_ptr(dest /*const void **/, pe /*int*/);
// NVSHMEM_PTR-NEXT: Is migrated to:
// NVSHMEM_PTR-NEXT:   local_ptr = ishmem_ptr(dest, pe);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=nvshmemx_init_attr | FileCheck %s -check-prefix=NVSHMEMX_INIT_ATTR
// NVSHMEMX_INIT_ATTR: CUDA API:
// NVSHMEMX_INIT_ATTR-NEXT:   nvshmemx_init_attr_t attributes;
// NVSHMEMX_INIT_ATTR-NEXT:   /* 1 */ nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM,
// NVSHMEMX_INIT_ATTR-NEXT:                              &attributes /*nvshmemx_init_attr_t **/);
// NVSHMEMX_INIT_ATTR-NEXT:   /* 2 */ nvshmemx_init_attr(NVSHMEMX_INIT_WITH_SHMEM,
// NVSHMEMX_INIT_ATTR-NEXT:                              &attributes /*nvshmemx_init_attr_t **/);
// NVSHMEMX_INIT_ATTR-NEXT: Is migrated to:
// NVSHMEMX_INIT_ATTR-NEXT:   ishmemx_attr_t attributes;
// NVSHMEMX_INIT_ATTR-NEXT:   /* 1 */ dpct::shmemx::init_attr(dpct::shmemx::RUNTIME_MPI, &attributes);
// NVSHMEMX_INIT_ATTR-NEXT:   /* 2 */ dpct::shmemx::init_attr(dpct::shmemx::RUNTIME_OPENSHMEM, &attributes);
