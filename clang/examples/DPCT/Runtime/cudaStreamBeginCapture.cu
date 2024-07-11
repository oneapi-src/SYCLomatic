// Option: --use-experimental-features=graph

void test(cudaStream_t s, cudaStreamCaptureMode sc) {
  // Start
  cudaStreamBeginCapture(s /*cudaStream_t*/, sc /*cudaStreamCaptureMode*/);
  // End
}
