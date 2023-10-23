__global__ void test(float f1, float f2, float f3) {
  // Start
  __fmaf_rd(f1 /*float*/, f2 /*float*/, f3 /*float*/);
  // End
}
