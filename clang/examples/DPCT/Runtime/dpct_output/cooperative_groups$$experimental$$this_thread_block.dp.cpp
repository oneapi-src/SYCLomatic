// Option: --use-experimental-features=logical-group
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

void test() {
  // Start
  auto cta =
      cooperative_groups::experimental::this_thread_block();
  // End
}
