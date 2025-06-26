// Option: --use-experimental-features=non-uniform-groups
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

__global__ void test() {
  // Start
   cooperative_groups::coalesced_group active = cooperative_groups::coalesced_threads();
  // End
}
