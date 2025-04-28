// Option: --use-experimental-features=bindless_images

void test(cudaExternalMemory_t *extMem,
          const cudaExternalMemoryHandleDesc *memHandleDesc) {
  // Start
  cudaImportExternalMemory(
      extMem /*cudaExternalMemory_t **/,
      memHandleDesc /*const cudaExternalMemoryHandleDesc **/);
  // End
}
