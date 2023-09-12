// Option: --use-experimental-features=free-function-queries
#include "cooperative_groups.h"
#include <cooperative_groups/reduce.h>

__global__ void test() {
  double *sdata;
  cooperative_groups::thread_block cta =
      cooperative_groups::this_thread_block();
  const unsigned int tid = cta.thread_rank();
  cooperative_groups::thread_block_tile<32> tile32 =
      cooperative_groups::tiled_partition<32>(cta);
  // Start
  cooperative_groups::reduce(
      tile32 /* type group */, sdata[tid] /* type argument */,
      cooperative_groups::plus<double>() /* type operator */);
  // End
}
