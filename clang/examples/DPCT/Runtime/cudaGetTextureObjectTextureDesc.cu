void test(cudaTextureDesc *pt) {
  // Start
  cudaTextureObject_t t;
  cudaGetTextureObjectTextureDesc(pt /*cudaTextureDesc **/,
                                  t /*cudaTextureObject_t*/);
  // End
}
