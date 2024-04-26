void test(const void *f, dim3 gridDim, dim3 blockDim, void **args,
          size_t sharedMem, cudaStream_t s) {
  // Start
  cudaLaunchKernel(f /*cudaError_t*/, gridDim /*dim3*/, blockDim /*dim3*/,
                   args /*void ***/, sharedMem /*size_t*/, s /*cudaStream_t*/);
  // End
}
