// Option: --use-experimental-features=bindless_images

void test(const cudaExternalSemaphore_t *extSemArray,
          const cudaExternalSemaphoreSignalParams *paramsArray,
          unsigned int numExtSems, cudaStream_t stream = 0) {
  // Start
  cudaSignalExternalSemaphoresAsync(
      extSemArray /*const cudaExternalSemaphore_t **/,
      paramsArray /*const cudaExternalSemaphoreSignalParams **/,
      numExtSems /*unsigned int*/);
  cudaSignalExternalSemaphoresAsync(
      extSemArray /*const cudaExternalSemaphore_t **/,
      paramsArray /*const cudaExternalSemaphoreSignalParams **/,
      numExtSems /*unsigned int*/, stream /*cudaStream_t*/);
  // End
}
