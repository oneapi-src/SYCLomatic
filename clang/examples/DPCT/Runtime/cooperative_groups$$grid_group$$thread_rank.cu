#include <cooperative_groups.h>

__global__ void test() {

  // Start
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  grid.thread_rank() /* grid_group::thread_rank */;
  // End
}
