// Option: --use-experimental-features=bindless_images

void test(cudaExternalSemaphore_t *extSem,
          const cudaExternalSemaphoreHandleDesc *semHandleDesc) {
  // Start
  cudaImportExternalSemaphore(
      extSem /*cudaExternalSemaphore_t **/,
      semHandleDesc /*const cudaExternalSemaphoreHandleDesc **/);
  // End
}
