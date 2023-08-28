void test(float *pf, cudaEvent_t e1, cudaEvent_t e2) {
  // Start
  cudaEventElapsedTime(pf /*float **/, e1 /*cudaEvent_t*/, e2 /*cudaEvent_t*/);
  // End
}
