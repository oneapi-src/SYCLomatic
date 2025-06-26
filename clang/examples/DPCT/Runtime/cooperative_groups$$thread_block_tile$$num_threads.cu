#include <cooperative_groups.h>

__global__ void test() {

  cooperative_groups::thread_block block =
      cooperative_groups::this_thread_block();

  // Start
  cooperative_groups::thread_block_tile<32> ctile32 =
      cooperative_groups::tiled_partition<32>(block);
  ctile32.num_threads();
  // End
}
