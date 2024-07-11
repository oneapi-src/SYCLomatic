// Option: --use-experimental-features=bindless_images

void test(cudaMipmappedArray_t m) {
  // Start
  cudaFreeMipmappedArray(m /*cudaMipmappedArray_t*/);
  // End
}
