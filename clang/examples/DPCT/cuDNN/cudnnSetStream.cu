#include <cudnn.h>

void test(cudnnHandle_t h, cudaStream_t s) {
  // Start
  cudnnSetStream(h /*cudnnHandle_t*/, s /*cudaStream_t*/);
  // End
}
