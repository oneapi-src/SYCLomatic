void test(cudaStream_t s, cudaStreamAttrID a, cudaStreamAttrValue *pv) {
  // Start
  cudaStreamSetAttribute(s /*cudaStream_t*/, a /*cudaStreamAttrID*/,
                         pv /*cudaStreamAttrValue **/);
  // End
}
