// Option: --use-experimental-features=bindless_images

void test(int c, cudaGraphicsResource_t *r, cudaStream_t s) {
  // Start
  cudaGraphicsUnmapResources(c /*int*/,
                             r /*cudaGraphicsResource_t **/);
  cudaGraphicsUnmapResources(c /*int*/,
                             r /*cudaGraphicsResource_t **/,
                             s /*cudaStream_t*/);
  // End
}
