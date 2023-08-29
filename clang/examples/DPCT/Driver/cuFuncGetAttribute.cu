void test(int *pi, CUfunction_attribute fa, CUfunction f) {
  // Start
  cuFuncGetAttribute(pi /*int **/, fa /*CUfunction_attribute*/,
                     f /*CUfunction*/);
  // End
}
