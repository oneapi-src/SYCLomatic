__global__ void test(unsigned long long int ull1, unsigned long long int ull2) {
  // Start
  __umul64hi(ull1 /*unsigned long long int*/, ull2 /*unsigned long long int*/);
  // End
}
