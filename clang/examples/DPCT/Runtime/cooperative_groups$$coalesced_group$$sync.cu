// Option: --use-experimental-features=non-uniform-groups
#include <cooperative_groups.h>

__global__ void test() {
  // Start
  cooperative_groups::coalesced_group active = cooperative_groups::coalesced_threads();
  active.sync();
  // End
}
