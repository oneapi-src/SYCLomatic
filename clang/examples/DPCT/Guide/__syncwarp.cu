// Option: --use-experimental-features=free-function-queries
__global__ void test(unsigned u) {
  // Start
  __syncwarp(u /*unsigned*/);
  // End
}
