// Option: --use-experimental-features=bindless_images

void test(cudaExternalSemaphore_t extSem) {
  // Start
  cudaDestroyExternalSemaphore(extSem /*cudaExternalSemaphore_t*/);
  // End
}
