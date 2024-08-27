// Option: --use-experimental-features=bindless_images

void test(cudaGraphicsResource_t r) {
  // Start
  cudaGraphicsUnregisterResource(r /*cudaGraphicsResource_t*/);
  // End
}
