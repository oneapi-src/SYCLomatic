// Option: --use-experimental-features=root-group
#include "cooperative_groups.h"
#include <cooperative_groups/reduce.h>

__global__ void test() {
  // Start
cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  // End
}
