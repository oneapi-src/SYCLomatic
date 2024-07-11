// Option: --use-experimental-features=masked-sub-group-operation
__device__ void test(unsigned int r, unsigned int mask, unsigned int var,
                     int src_lane, int width) {
  // Start
  r = __shfl_sync(mask /*unsigned int*/, var /*unsigned int*/, src_lane /*int*/,
                  width /*int*/);
  // End
}