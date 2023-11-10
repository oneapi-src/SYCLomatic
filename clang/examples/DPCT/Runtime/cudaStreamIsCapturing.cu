void test(cudaStream_t s, enum cudaStreamCaptureStatus *ps) {
  // Start
  cudaStreamIsCapturing(s /*cudaStream_t*/,
                        ps /* enum cudaStreamCaptureStatus **/);
  // End
}
