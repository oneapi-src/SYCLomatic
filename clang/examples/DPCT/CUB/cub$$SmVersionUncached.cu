// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

void test(int res) {
  // Start
  cub::SmVersionUncached(res/*int*/);
  // End
}
