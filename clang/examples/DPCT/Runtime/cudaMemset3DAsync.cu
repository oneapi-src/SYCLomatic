void test(int i) {
  // Start
  cudaPitchedPtr p;
  cudaExtent e;
  cudaStream_t s;
  cudaMemset3DAsync(p, i /*int*/, e, s);
  // End
}
