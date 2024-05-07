void test(cudaStream_t s, cudaStreamAttrID a, cudaStreamAttrValue *pv) {
  // Start
  cudaStreamGetAttribute(s /*cudaStream_t*/, a /*cudaStreamAttrID*/,
                         pv /*cudaStreamAttrValue **/);
  // End
}
