// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaImportExternalMemory | FileCheck %s -check-prefix=CUDA_IMPORT_EXTERNAL_MEMORY
// CUDA_IMPORT_EXTERNAL_MEMORY: CUDA API:
// CUDA_IMPORT_EXTERNAL_MEMORY-NEXT:    cudaImportExternalMemory(
// CUDA_IMPORT_EXTERNAL_MEMORY-NEXT:        extMem /*cudaExternalMemory_t **/,
// CUDA_IMPORT_EXTERNAL_MEMORY-NEXT:        memHandleDesc /*const cudaExternalMemoryHandleDesc **/);
// CUDA_IMPORT_EXTERNAL_MEMORY-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// CUDA_IMPORT_EXTERNAL_MEMORY-NEXT:    dpct::experimental::import_external_memory(extMem, memHandleDesc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaExternalMemoryGetMappedBuffer | FileCheck %s -check-prefix=CUDA_EXTERNAL_MEMORY_GET_MAPPED_BUFFER
// CUDA_EXTERNAL_MEMORY_GET_MAPPED_BUFFER: CUDA API:
// CUDA_EXTERNAL_MEMORY_GET_MAPPED_BUFFER-NEXT:    cudaExternalMemoryGetMappedBuffer(
// CUDA_EXTERNAL_MEMORY_GET_MAPPED_BUFFER-NEXT:        devPtr /*void ***/, extMem /*cudaExternalMemory_t*/,
// CUDA_EXTERNAL_MEMORY_GET_MAPPED_BUFFER-NEXT:        bufferDesc /*const cudaExternalMemoryBufferDesc **/);
// CUDA_EXTERNAL_MEMORY_GET_MAPPED_BUFFER-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// CUDA_EXTERNAL_MEMORY_GET_MAPPED_BUFFER-NEXT:    *devPtr = sycl::ext::oneapi::experimental::map_external_linear_memory(extMem, bufferDesc->get_res_size(), bufferDesc->get_mem_offset(), dpct::get_in_order_queue());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaExternalMemoryGetMappedMipmappedArray | FileCheck %s -check-prefix=CUDA_EXTERNAL_MEMORY_GET_MAPPED_MIPMAPPED_ARRAY
// CUDA_EXTERNAL_MEMORY_GET_MAPPED_MIPMAPPED_ARRAY: CUDA API:
// CUDA_EXTERNAL_MEMORY_GET_MAPPED_MIPMAPPED_ARRAY-NEXT:    cudaExternalMemoryGetMappedMipmappedArray(
// CUDA_EXTERNAL_MEMORY_GET_MAPPED_MIPMAPPED_ARRAY-NEXT:        mipmap /*cudaMipmappedArray_t **/, extMem /*cudaExternalMemory_t*/,
// CUDA_EXTERNAL_MEMORY_GET_MAPPED_MIPMAPPED_ARRAY-NEXT:        mipmapDesc /*const cudaExternalMemoryMipmappedArrayDesc **/);
// CUDA_EXTERNAL_MEMORY_GET_MAPPED_MIPMAPPED_ARRAY-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// CUDA_EXTERNAL_MEMORY_GET_MAPPED_MIPMAPPED_ARRAY-NEXT:    mipmap = new dpct::experimental::image_mem_wrapper(extMem, mipmapDesc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaDestroyExternalMemory | FileCheck %s -check-prefix=CUDA_DESTROY_EXTERNAL_MEMORY
// CUDA_DESTROY_EXTERNAL_MEMORY: CUDA API:
// CUDA_DESTROY_EXTERNAL_MEMORY-NEXT:    cudaDestroyExternalMemory(extMem /*cudaExternalMemory_t*/);
// CUDA_DESTROY_EXTERNAL_MEMORY-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// CUDA_DESTROY_EXTERNAL_MEMORY-NEXT:    sycl::ext::oneapi::experimental::release_external_memory(extMem, dpct::get_in_order_queue());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaImportExternalSemaphore | FileCheck %s -check-prefix=CUDA_IMPORT_EXTERNAL_SEMAPHORE
// CUDA_IMPORT_EXTERNAL_SEMAPHORE: CUDA API:
// CUDA_IMPORT_EXTERNAL_SEMAPHORE-NEXT:    cudaImportExternalSemaphore(
// CUDA_IMPORT_EXTERNAL_SEMAPHORE-NEXT:        extSem /*cudaExternalSemaphore_t **/,
// CUDA_IMPORT_EXTERNAL_SEMAPHORE-NEXT:        semHandleDesc /*const cudaExternalSemaphoreHandleDesc **/);
// CUDA_IMPORT_EXTERNAL_SEMAPHORE-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// CUDA_IMPORT_EXTERNAL_SEMAPHORE-NEXT:    dpct::experimental::import_external_semaphore(extSem, semHandleDesc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaSignalExternalSemaphoresAsync | FileCheck %s -check-prefix=CUDA_SIGNAL_EXTERNAL_SEMAPHORES_ASYNC
// CUDA_SIGNAL_EXTERNAL_SEMAPHORES_ASYNC: CUDA API:
// CUDA_SIGNAL_EXTERNAL_SEMAPHORES_ASYNC-NEXT:    cudaSignalExternalSemaphoresAsync(
// CUDA_SIGNAL_EXTERNAL_SEMAPHORES_ASYNC-NEXT:        extSemArray /*const cudaExternalSemaphore_t **/,
// CUDA_SIGNAL_EXTERNAL_SEMAPHORES_ASYNC-NEXT:        paramsArray /*const cudaExternalSemaphoreSignalParams **/,
// CUDA_SIGNAL_EXTERNAL_SEMAPHORES_ASYNC-NEXT:        numExtSems /*unsigned int*/);
// CUDA_SIGNAL_EXTERNAL_SEMAPHORES_ASYNC-NEXT:    cudaSignalExternalSemaphoresAsync(
// CUDA_SIGNAL_EXTERNAL_SEMAPHORES_ASYNC-NEXT:        extSemArray /*const cudaExternalSemaphore_t **/,
// CUDA_SIGNAL_EXTERNAL_SEMAPHORES_ASYNC-NEXT:        paramsArray /*const cudaExternalSemaphoreSignalParams **/,
// CUDA_SIGNAL_EXTERNAL_SEMAPHORES_ASYNC-NEXT:        numExtSems /*unsigned int*/, stream /*cudaStream_t*/);
// CUDA_SIGNAL_EXTERNAL_SEMAPHORES_ASYNC-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// CUDA_SIGNAL_EXTERNAL_SEMAPHORES_ASYNC-NEXT:    dpct::experimental::signal_external_semaphores(extSemArray, paramsArray, numExtSems);
// CUDA_SIGNAL_EXTERNAL_SEMAPHORES_ASYNC-NEXT:    dpct::experimental::signal_external_semaphores(extSemArray, paramsArray, numExtSems, stream);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaWaitExternalSemaphoresAsync | FileCheck %s -check-prefix=CUDA_WAIT_EXTERNAL_SEMAPHORES_ASYNC
// CUDA_WAIT_EXTERNAL_SEMAPHORES_ASYNC: CUDA API:
// CUDA_WAIT_EXTERNAL_SEMAPHORES_ASYNC-NEXT:    cudaWaitExternalSemaphoresAsync(
// CUDA_WAIT_EXTERNAL_SEMAPHORES_ASYNC-NEXT:        extSemArray /*const cudaExternalSemaphore_t **/,
// CUDA_WAIT_EXTERNAL_SEMAPHORES_ASYNC-NEXT:        paramsArray /*const cudaExternalSemaphoreWaitParams **/,
// CUDA_WAIT_EXTERNAL_SEMAPHORES_ASYNC-NEXT:        numExtSems /*unsigned int*/);
// CUDA_WAIT_EXTERNAL_SEMAPHORES_ASYNC-NEXT:    cudaWaitExternalSemaphoresAsync(
// CUDA_WAIT_EXTERNAL_SEMAPHORES_ASYNC-NEXT:        extSemArray /*const cudaExternalSemaphore_t **/,
// CUDA_WAIT_EXTERNAL_SEMAPHORES_ASYNC-NEXT:        paramsArray /*const cudaExternalSemaphoreWaitParams **/,
// CUDA_WAIT_EXTERNAL_SEMAPHORES_ASYNC-NEXT:        numExtSems /*unsigned int*/, stream /*cudaStream_t*/);
// CUDA_WAIT_EXTERNAL_SEMAPHORES_ASYNC-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// CUDA_WAIT_EXTERNAL_SEMAPHORES_ASYNC-NEXT:    dpct::experimental::wait_external_semaphores(extSemArray, paramsArray, numExtSems);
// CUDA_WAIT_EXTERNAL_SEMAPHORES_ASYNC-NEXT:    dpct::experimental::wait_external_semaphores(extSemArray, paramsArray, numExtSems, stream);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaDestroyExternalSemaphore | FileCheck %s -check-prefix=CUDA_DESTROY_EXTERNAL_SEMAPHORE
// CUDA_DESTROY_EXTERNAL_SEMAPHORE: CUDA API:
// CUDA_DESTROY_EXTERNAL_SEMAPHORE-NEXT:    cudaDestroyExternalSemaphore(extSem /*cudaExternalSemaphore_t*/);
// CUDA_DESTROY_EXTERNAL_SEMAPHORE-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// CUDA_DESTROY_EXTERNAL_SEMAPHORE-NEXT:    delete extSem;
