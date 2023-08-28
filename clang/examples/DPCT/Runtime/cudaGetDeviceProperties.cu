void test(int i) {
  // Start
  cudaDeviceProp *pd;
  cudaGetDeviceProperties(pd, i /*int*/);
  // End
}
