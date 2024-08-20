void test(CUfunction f, CUfunction_attribute fa, int i) {
  // Start
  cuFuncSetAttribute(f /*CUfunction*/, fa /*CUfunction_attribute*/, i /*int*/);
  // End
}
