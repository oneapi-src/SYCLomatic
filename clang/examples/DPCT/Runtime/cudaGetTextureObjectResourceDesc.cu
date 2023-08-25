void test(cudaResourceDesc *pr) {
  // Start
  cudaTextureObject_t t;
  cudaGetTextureObjectResourceDesc(pr /*cudaResourceDesc **/,
                                   t /*cudaTextureObject_t*/);
  // End
}
