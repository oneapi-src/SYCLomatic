void test(int i1, int i2, int i3, int i4, cudaChannelFormatKind c) {
  // Start
  cudaCreateChannelDesc(i1 /*int*/, i2 /*int*/, i3 /*int*/, i4 /*int*/,
                        c /*cudaChannelFormatKind*/);
  // End
}
