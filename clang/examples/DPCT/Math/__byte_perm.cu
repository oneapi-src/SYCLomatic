__global__ void test(unsigned int u1, unsigned int u2, unsigned int u3) {
  // Start
  __byte_perm(u1 /*unsigned int*/, u2 /*unsigned int*/, u3 /*unsigned int*/);
  // End
}
