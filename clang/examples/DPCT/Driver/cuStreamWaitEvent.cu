void test(CUevent e, unsigned int u) {
  // Start
  CUstream s;
  cuStreamWaitEvent(s /*CUstream*/, e /*CUevent*/, u /*unsigned int*/);
  // End
}
