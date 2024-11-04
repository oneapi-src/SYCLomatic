__global__ void f() {}

void test() {
  dim3 gridDim, blockDim;
  // Start
  f<<<gridDim, blockDim>>>();
  // End
}
