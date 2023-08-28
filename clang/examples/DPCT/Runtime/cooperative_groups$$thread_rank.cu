// Option: --use-experimental-features=free-function-queries
#include "cooperative_groups.h"
#include <cooperative_groups/reduce.h>


__global__ void test() {
    // Start
    cooperative_groups::thread_block cta = cooperative_groups::this_thread_block();
    cta.thread_rank();
    // End
}


