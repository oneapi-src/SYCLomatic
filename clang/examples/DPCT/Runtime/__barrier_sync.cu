__global__ void test(unsigned id) {
  goto label;
  // Start
  __barrier_sync(id /*unsigned*/);
  // End
label:
  int a;
}
