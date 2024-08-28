// Option: --use-experimental-features=bindless_images

void test(int c, cudaGraphicsResource_t *r, cudaStream_t s) {
  // Start
  cudaGraphicsMapResources(c /*int*/,
                           r /*cudaGraphicsResource_t **/);
  cudaGraphicsMapResources(c /*int*/,
                           r /*cudaGraphicsResource_t **/,
                           s /*cudaStream_t*/);
  // End
}
