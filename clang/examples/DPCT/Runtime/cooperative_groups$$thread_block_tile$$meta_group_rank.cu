#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>

__global__ void test() {

  cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
  cooperative_groups::thread_block_tile<32> ctile32 = cooperative_groups::tiled_partition<32>(block);

  // Start
  ctile32.meta_group_rank();// thread_block_tile<tile size>::meta_group_rank
  // End
}
