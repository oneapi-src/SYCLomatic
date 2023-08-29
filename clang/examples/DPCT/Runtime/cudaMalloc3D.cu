void test(cudaPitchedPtr *pp) {
  // Start
  cudaExtent e;
  cudaMalloc3D(pp /*cudaPitchedPtr **/, e);
  // End
}
