// Option: --enable-profiling

void test(cudaEvent_t event, cudaStream_t stream) {
  // Start
  cudaEventRecordWithFlags(event /*cudaEvent_t*/, stream /*cudaStream_t*/,
                           cudaEventRecordDefault /*unsigned int*/);
  // End
}
