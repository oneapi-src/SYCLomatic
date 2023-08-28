void test(CUevent e) {
  // Start
  CUstream s;
  cuEventRecord(e /*CUevent*/, s /*CUstream*/);
  // End
}
