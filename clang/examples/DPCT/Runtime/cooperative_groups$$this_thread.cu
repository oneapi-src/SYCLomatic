// Option: --use-experimental-features=logical-group
#include "cooperative_groups.h"
#include <cooperative_groups/reduce.h>

__global__ void test() {
  // Start
  auto thread = cooperative_groups::this_thread();
  // End
}
