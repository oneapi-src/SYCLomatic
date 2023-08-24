void test(CUfunction *pf, CUmodule m, const char *pc) {
  // Start
  cuModuleGetFunction(pf /*CUfunction **/, m /*CUmodule*/, pc /*const char **/);
  // End
}
