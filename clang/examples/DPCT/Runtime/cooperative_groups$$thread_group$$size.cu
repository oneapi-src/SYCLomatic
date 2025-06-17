#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>

__global__ void test() {

  cooperative_groups::thread_group tg =
      cooperative_groups::this_thread_block();

  // Start
    tg.size()/* thread_group::size */;
  // End
}
