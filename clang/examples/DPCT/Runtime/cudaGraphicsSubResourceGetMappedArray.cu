// Option: --use-experimental-features=bindless_images

void test(cudaArray_t a, cudaGraphicsResource_t r, unsigned i, unsigned l) {
  // Start
  cudaGraphicsSubResourceGetMappedArray(&a /*cudaArray_t **/,
                                        r /*cudaGraphicsResource_t*/,
                                        i /*unsigned*/,
                                        l /*unsigned*/);
  // End
}
