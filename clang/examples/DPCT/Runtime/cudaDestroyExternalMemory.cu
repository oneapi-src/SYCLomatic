// Option: --use-experimental-features=bindless_images

void test(cudaExternalMemory_t extMem) {
  // Start
  cudaDestroyExternalMemory(extMem /*cudaExternalMemory_t*/);
  // End
}
