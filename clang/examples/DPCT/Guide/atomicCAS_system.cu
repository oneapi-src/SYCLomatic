__global__ void test(int *pi, int i1, int i2, unsigned *pu, unsigned u1,
                     unsigned u2, unsigned long long *pull,
                     unsigned long long ull1, unsigned long long ull2,
                     unsigned short *pus, unsigned short us1,
                     unsigned short us2) {
  // Start
  atomicCAS_system(pi /*int **/, i1 /*int*/, i2 /*int*/);
  atomicCAS_system(pu /*unsigned **/, u1 /*unsigned*/, u2 /*unsigned*/);
  atomicCAS_system(pull /*unsigned long long **/, ull1 /*unsigned long long*/,
                   ull2 /*unsigned long long*/);
  // End
}
