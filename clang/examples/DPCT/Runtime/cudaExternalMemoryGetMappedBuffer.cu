// Option: --use-experimental-features=bindless_images

void test(void **devPtr, cudaExternalMemory_t extMem,
          const cudaExternalMemoryBufferDesc *bufferDesc) {
  // Start
  cudaExternalMemoryGetMappedBuffer(
      devPtr /*void ***/, extMem /*cudaExternalMemory_t*/,
      bufferDesc /*const cudaExternalMemoryBufferDesc **/);
  // End
}
