#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>

__global__ void test() {

  // Start
  cooperative_groups::thread_block block =
      cooperative_groups::this_thread_block();
  cooperative_groups::thread_block_tile<32> ctile32 =
      cooperative_groups::tiled_partition<32>(block);
  cooperative_groups::thread_block_tile<16> ctile16 =
      cooperative_groups::tiled_partition<16>(block);
  ctile32.thread_rank();
  ctile16.thread_rank();
  // End
}
