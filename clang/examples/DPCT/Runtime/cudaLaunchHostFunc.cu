void test(cudaStream_t stream, cudaHostFn_t fn, void*userData) {
  // Start
  cudaLaunchHostFunc(stream/*cudaStream_t*/, fn/*cudaHostFn_t*/, userData/*void**/);
  // End
}
