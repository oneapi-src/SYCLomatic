void test(const void *f, cudaFuncAttribute attr, int i) {
  // Start
  cudaFuncSetAttribute(f /*const void **/, attr /*cudaFuncAttribute*/,
                       i /*int*/);
  // End
}
