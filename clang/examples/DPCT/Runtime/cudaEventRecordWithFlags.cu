void test(cudaEvent_t event, cudaStream_t stream, unsigned int flags) {
  // Start
  cudaEventRecordWithFlags(event/*cudaEvent_t*/, stream/*cudaStream_t*/, flags/*unsigned int*/);
  // End
}
