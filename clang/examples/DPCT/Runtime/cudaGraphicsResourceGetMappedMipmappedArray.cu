// Option: --use-experimental-features=bindless_images

void test(cudaMipmappedArray_t m, cudaGraphicsResource_t r) {
  // Start
  cudaGraphicsResourceGetMappedMipmappedArray(&m /*cudaMipmappedArray_t **/,
                                              r /*cudaGraphicsResource_t*/);
  // End
}
