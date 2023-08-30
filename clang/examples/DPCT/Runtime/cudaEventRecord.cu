void test(cudaEvent_t e) {
  // Start
  cudaStream_t s;
  cudaEventRecord(e /*cudaEvent_t*/, s /*cudaStream_t*/);
  // End
}
