#include "cooperative_groups.h"
#include <cooperative_groups/reduce.h>


__global__ void test(int *sdata, const cooperative_groups::thread_block &cta) {
    // Start
    const unsigned int tid = cta.thread_rank();
    cooperative_groups::thread_block_tile<32> tile32 = cooperative_groups::tiled_partition<32>(cta);
    cooperative_groups::reduce(tile32 /* cooperative_groups::thread_block_til */, sdata[tid] /* void */, cooperative_groups::less<int>() /* */);
    // End
}
