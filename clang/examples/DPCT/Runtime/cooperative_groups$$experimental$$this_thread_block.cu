#define _CG_ABI_EXPERIMENTAL

#include <cooperative_groups.h>
__device__
void _Copy() {
  __shared__ cooperative_groups::experimental::block_tile_memory<8> shared;
// Start
 cooperative_groups::thread_block thb = cooperative_groups::experimental::this_thread_block(shared/*cooperative_groups::experimental::block_tile_memory*/);
// End
}

