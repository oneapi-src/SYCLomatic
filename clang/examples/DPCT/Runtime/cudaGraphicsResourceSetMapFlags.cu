// Option: --use-experimental-features=bindless_images

void test(cudaGraphicsResource_t r, unsigned f) {
  // Start
  cudaGraphicsResourceSetMapFlags(r /*cudaGraphicsResource_t*/,
                                  f /*unsigned*/);
  // End
}
