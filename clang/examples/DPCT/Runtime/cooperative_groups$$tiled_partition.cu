// Option: --use-experimental-features=logical-group
// Option: --use-experimental-features=free-function-queries
#include "cooperative_groups.h"
#include <cooperative_groups/reduce.h>


__global__ void test() {
    // Start
    cooperative_groups::thread_block cta = cooperative_groups::this_thread_block();
    cooperative_groups::tiled_partition<32>(cta);
    cooperative_groups::tiled_partition<16>(cta);
    // End
}

