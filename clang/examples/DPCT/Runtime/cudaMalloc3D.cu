void test(cudaPitchedPtr *pitch) {
  // Start
  cudaExtent e;
  cudaMalloc3D(pitch /*cudaPitchedPtr **/, e);
  // End
}
