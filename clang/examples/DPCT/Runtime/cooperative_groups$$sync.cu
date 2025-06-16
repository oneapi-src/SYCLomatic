// Option: --use-experimental-features=logical-group 
#include "cooperative_groups.h"
#include <cooperative_groups/reduce.h>


__global__ void test() {
    cooperative_groups::thread_block tb = cooperative_groups::this_thread_block();
  cooperative_groups::thread_block_tile<32> tbt32 = cooperative_groups::tiled_partition<32>(tb);
  // Start
  tb/*thread_block*/.sync();
  tbt32/*thread_block_tile<32>*/.sync();
  cooperative_groups::sync(tb/*thread_block*/);
  cooperative_groups::sync(tbt32/*thread_block_tile<32>*/);
  // End
}
