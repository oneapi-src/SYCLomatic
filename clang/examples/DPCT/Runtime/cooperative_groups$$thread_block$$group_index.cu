#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>

__global__ void test() {

  cooperative_groups::thread_block tb = cooperative_groups::this_thread_block();

  // Start
  tb.thread_index() /* thread_block::thread_index */;
  // End
}
