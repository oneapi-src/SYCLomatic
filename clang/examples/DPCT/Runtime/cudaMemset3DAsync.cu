void test(cudaPitchedPtr p, int i, cudaExtent e, cudaStream_t s) {
  // Start
  cudaMemset3DAsync(p /*cudaPitchedPtr*/, i /*int*/, e /*cudaExtent*/,
                    s /*cudaStream_t*/);
  // End
}
