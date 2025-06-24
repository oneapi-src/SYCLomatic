// UNSUPPORTED: system-windows
// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaIpcOpenMemHandle | FileCheck %s -check-prefix=CUDA_IPC_OPEN_MEM_HANDLE
// CUDA_IPC_OPEN_MEM_HANDLE: CUDA API:
// CUDA_IPC_OPEN_MEM_HANDLE-NEXT:  cudaIpcOpenMemHandle((void **)&ptr/*void ***/, *handle/*cudaIpcMemHandle_t*/, cudaIpcMemLazyEnablePeerAccess/*unsigned int*/);
// CUDA_IPC_OPEN_MEM_HANDLE-NEXT: Is migrated to (with the option --use-experimental-features=level_zero):
// CUDA_IPC_OPEN_MEM_HANDLE-NEXT:  dpct::experimental::open_mem_ipc_handle(*handle, (void **)&ptr);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaIpcCloseMemHandle | FileCheck %s -check-prefix=CUDA_IPC_CLOSE_MEM_HANDLE
// CUDA_IPC_CLOSE_MEM_HANDLE: CUDA API:
// CUDA_IPC_CLOSE_MEM_HANDLE-NEXT:   cudaIpcCloseMemHandle(ptr/*void **/);
// CUDA_IPC_CLOSE_MEM_HANDLE-NEXT: Is migrated to (with the option --use-experimental-features=level_zero):
// CUDA_IPC_CLOSE_MEM_HANDLE-NEXT:  zeMemCloseIpcHandle(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dpct::get_current_device().get_context()), ptr);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaIpcGetMemHandle | FileCheck %s -check-prefix=CUDA_IPC_GET_MEM_HANDLE
// CUDA_IPC_GET_MEM_HANDLE: CUDA API:
// CUDA_IPC_GET_MEM_HANDLE-NEXT:  cudaIpcGetMemHandle(handle/*cudaIpcMemHandle_t **/, ptr/*void **/)
// CUDA_IPC_GET_MEM_HANDLE-NEXT: Is migrated to (with the option --use-experimental-features=level_zero):
// CUDA_IPC_GET_MEM_HANDLE-NEXT:  dpct::experimental::get_mem_ipc_handle(ptr, handle);