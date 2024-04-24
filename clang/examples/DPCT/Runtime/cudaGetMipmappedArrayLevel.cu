// Option: --use-experimental-features=bindless_images

void test(cudaArray_t *a, const cudaMipmappedArray_t m, unsigned u) {
  // Start
  cudaGetMipmappedArrayLevel(a /*cudaArray_t **/,
                             m /*const cudaMipmappedArray_t*/, u /*unsigned*/);
  // End
}
