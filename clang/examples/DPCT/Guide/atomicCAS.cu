__global__ void test(int *pi, int i1, int i2, unsigned *pu, unsigned u1,
                     unsigned u2, unsigned long long *pull,
                     unsigned long long ull1, unsigned long long ull2) {
  // Start
  atomicCAS(pi /*int **/, i1 /*int*/, i2 /*int*/);
  atomicCAS(pu /*unsigned **/, u1 /*unsigned*/, u2 /*unsigned*/);
  atomicCAS(pull /*unsigned long long **/, ull1 /*unsigned long long*/,
            ull2 /*unsigned long long*/);
  // End
}
