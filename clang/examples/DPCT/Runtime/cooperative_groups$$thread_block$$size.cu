#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>

__global__ void test() {
  // Start
  cooperative_groups::thread_block tb = cooperative_groups::this_thread_block();
  tb.size();
  // End
}
