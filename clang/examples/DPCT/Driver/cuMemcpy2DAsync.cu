void test(const CUDA_MEMCPY2D *pm, CUstream s) {
  // Start
  cuMemcpy2DAsync(pm /*const CUDA_MEMCPY2D **/, s /*CUstream*/);
  // End
}
