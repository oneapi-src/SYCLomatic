void test(cudaPitchedPtr p, int i, cudaExtent e) {
  // Start
  cudaMemset3D(p /*cudaPitchedPtr*/, i /*int*/, e /*cudaExtent*/);
  // End
}
