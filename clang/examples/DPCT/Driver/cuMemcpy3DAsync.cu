void test(const CUDA_MEMCPY3D *pm, CUstream s) {
  // Start
  cuMemcpy3DAsync(pm /*const CUDA_MEMCPY3D **/, s /*CUstream*/);
  // End
}
