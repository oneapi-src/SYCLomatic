// Option: --use-experimental-features=logical-group 
#include "cooperative_groups.h"
#include <cooperative_groups/reduce.h>

__global__ void test() {

  // Start
  cooperative_groups::thread_block tb = cooperative_groups::this_thread_block();
  cooperative_groups::thread_block_tile<32> tbt32 =
      cooperative_groups::tiled_partition<32>(tb);
  tb.sync();
  tbt32.sync();
  cooperative_groups::sync(tb);
  cooperative_groups::sync(tbt32);
  // End
}
