#include "cooperative_groups.h"
#include <cooperative_groups/reduce.h>


__global__ void test(const cooperative_groups::thread_block &cta) {
    // Start
    cta.thread_rank();
    // End
}


