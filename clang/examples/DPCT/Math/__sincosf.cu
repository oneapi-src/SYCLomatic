__global__ void test(float f, float *pf1, float *pf2) {
  // Start
  __sincosf(f /*float*/, pf1 /*float **/, pf2 /*float **/);
  // End
}
