// REQUIRES: system-linux
// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=nvshmem_align | FileCheck %s -check-prefix=NVSHMEM_ALIGN
// NVSHMEM_ALIGN: CUDA API:
// NVSHMEM_ALIGN-NEXT:   nvshmem_align(alignment /*size_t*/, size /*size_t*/);
// NVSHMEM_ALIGN-NEXT: Is migrated to:
// NVSHMEM_ALIGN-NEXT:   ishmem_align(alignment, size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=nvshmem_calloc | FileCheck %s -check-prefix=NVSHMEM_CALLOC
// NVSHMEM_CALLOC: CUDA API:
// NVSHMEM_CALLOC-NEXT:   nvshmem_calloc(count /*size_t*/, size /*size_t*/);
// NVSHMEM_CALLOC-NEXT: Is migrated to:
// NVSHMEM_CALLOC-NEXT:   ishmem_calloc(count, size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=nvshmem_free | FileCheck %s -check-prefix=NVSHMEM_FREE
// NVSHMEM_FREE: CUDA API:
// NVSHMEM_FREE-NEXT:   nvshmem_free(ptr /*void **/);
// NVSHMEM_FREE-NEXT: Is migrated to:
// NVSHMEM_FREE-NEXT:   ishmem_free(ptr);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=nvshmem_malloc | FileCheck %s -check-prefix=NVSHMEM_MALLOC
// NVSHMEM_MALLOC: CUDA API:
// NVSHMEM_MALLOC-NEXT:   nvshmem_malloc(size /*size_t*/);
// NVSHMEM_MALLOC-NEXT: Is migrated to:
// NVSHMEM_MALLOC-NEXT:   ishmem_malloc(size);
