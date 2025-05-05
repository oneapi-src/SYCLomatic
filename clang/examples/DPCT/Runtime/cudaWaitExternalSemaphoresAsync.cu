// Option: --use-experimental-features=bindless_images

void test(const cudaExternalSemaphore_t *extSemArray,
          const cudaExternalSemaphoreWaitParams *paramsArray,
          unsigned int numExtSems, cudaStream_t stream = 0) {
  // Start
  cudaWaitExternalSemaphoresAsync(
      extSemArray /*const cudaExternalSemaphore_t **/,
      paramsArray /*const cudaExternalSemaphoreWaitParams **/,
      numExtSems /*unsigned int*/);
  cudaWaitExternalSemaphoresAsync(
      extSemArray /*const cudaExternalSemaphore_t **/,
      paramsArray /*const cudaExternalSemaphoreWaitParams **/,
      numExtSems /*unsigned int*/, stream /*cudaStream_t*/);
  // End
}
