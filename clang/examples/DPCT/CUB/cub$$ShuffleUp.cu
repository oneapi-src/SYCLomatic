// Option: --use-experimental-features=non-uniform-groups
// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

__device__ void test(int output, int input, int src_offset, int first_thread, unsigned int member_mask) {
  // Start
  output /*int*/ = cub::ShuffleUp<32>(input /*int*/, src_offset /*int*/, first_thread /*int*/, member_mask /*unsigned int*/);
  // End
}
