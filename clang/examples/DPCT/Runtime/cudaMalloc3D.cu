void test(cudaPitchedPtr *pp, cudaExtent e) {
  // Start
  cudaMalloc3D(pp /*cudaPitchedPtr **/, e /*cudaExtent*/);
  // End
}
