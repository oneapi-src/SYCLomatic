void test(cudaEvent_t e, unsigned int u) {
  // Start
  cudaStream_t s;
  cudaStreamWaitEvent(s /*cudaStream_t*/, e /*cudaEvent_t*/,
                      u /*unsigned int*/);
  // End
}
