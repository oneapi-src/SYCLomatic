__global__ void test(float f, float *pf) {
  // Start
  modff(f /*float*/, pf /*float **/);
  // End
}
