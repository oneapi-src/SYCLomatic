void test(CUstream s, CUevent e, unsigned int u) {
  // Start
  cuStreamWaitEvent(s /*CUstream*/, e /*CUevent*/, u /*unsigned int*/);
  // End
}
