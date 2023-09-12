void test(CUfunction f, unsigned int u1, unsigned int u2, unsigned int u3,
          unsigned int u4, unsigned int u5, unsigned int u6, unsigned int u7,
          CUstream s, void **pParam, void **pOpt) {
  // Start
  cuLaunchKernel(f /*CUfunction*/, u1 /*unsigned int*/, u2 /*unsigned int*/,
                 u3 /*unsigned int*/, u4 /*unsigned int*/, u5 /*unsigned int*/,
                 u6 /*unsigned int*/, u7 /*unsigned int*/, s /*CUstream*/,
                 pParam /*void ***/, pOpt /*void ***/);
  // End
}
