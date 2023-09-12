void test(int i) {
  // Start
  cudaPitchedPtr p;
  cudaExtent e;
  cudaMemset3D(p, i /*int*/, e);
  // End
}
