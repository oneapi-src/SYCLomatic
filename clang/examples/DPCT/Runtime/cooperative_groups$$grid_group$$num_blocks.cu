#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>

__global__ void test() {

  cooperative_groups::grid_group grid = cooperative_groups::this_grid();

  // Start
  grid.num_blocks() /* grid_group::num_blocks */;
  // End
}
