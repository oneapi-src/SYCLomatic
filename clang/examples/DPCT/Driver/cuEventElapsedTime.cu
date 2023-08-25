void test(float *pf, CUevent e1, CUevent e2) {
  // Start
  cuEventElapsedTime(pf /*float **/, e1 /*CUevent*/, e2 /*CUevent*/);
  // End
}
