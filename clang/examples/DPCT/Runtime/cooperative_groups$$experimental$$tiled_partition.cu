
#define _CG_ABI_EXPERIMENTAL
#include <cuda.h>
#include "cooperative_groups.h"

__device__ void test(cooperative_groups::thread_block tb) {
  cooperative_groups::experimental::block_tile_memory<1, 1> mem;
  // Start
    cooperative_groups::thread_block_tile<32> tbt32 = cooperative_groups::experimental::tiled_partition<32>(tb/*cooperative_groups::thread_block*/);
  // End
}
