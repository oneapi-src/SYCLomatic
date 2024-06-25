// Option: --use-experimental-features=graph

void test(cudaStream_t s) {
  // Start
  cudaStreamBeginCapture(s /*cudaStream_t*/, cudaStreamCaptureModeGlobal);
  // End
}
