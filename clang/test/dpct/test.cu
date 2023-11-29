#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;


__device__ void reduceBlock(double *sdata, const cg::thread_block &cta) {
  const unsigned int tid = cta.thread_rank();
  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  cg::reduce(tile32, sdata[tid], cg::plus<double>());  
  cg::sync(cta);

}
