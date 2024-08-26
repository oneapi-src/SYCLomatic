// Option: --use-experimental-features=bindless_images

void test(void *ptr, size_t *s, cudaGraphicsResource_t r) {
  // Start
  cudaGraphicsResourceGetMappedPointer(&ptr /*void ***/,
                                       s /*size_t **/,
                                       r /*cudaGraphicsResource_t*/);
  // End
}
