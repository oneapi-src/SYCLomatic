// Option: --use-experimental-features=bindless_images

void test(cudaMipmappedArray_t *m, const cudaChannelFormatDesc *d, cudaExtent e,
          unsigned u1, unsigned u2) {
  // Start
  cudaMallocMipmappedArray(m /*cudaMipmappedArray_t **/,
                           d /*const cudaChannelFormatDesc **/,
                           e /*cudaExtent*/, u1 /*unsigned*/, u2 /*unsigned*/);
  // End
}
