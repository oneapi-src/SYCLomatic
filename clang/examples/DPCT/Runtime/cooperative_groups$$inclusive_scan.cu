#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>

__global__ void test() {
  double *sdata;
  cooperative_groups::thread_block cta =
      cooperative_groups::this_thread_block();
  const unsigned int tid = cta.thread_rank();
  cooperative_groups::thread_block_tile<32> tile32 =
      cooperative_groups::tiled_partition<32>(cta);

  // Start
  cooperative_groups::inclusive_scan(
      tile32 /* type group */, sdata[tid] /* type value */,
      cooperative_groups::plus<double>() /* type operator */);

  cooperative_groups::inclusive_scan(tile32 /* type group */,
                                     sdata[tid] /* type value */);

  // End
}
