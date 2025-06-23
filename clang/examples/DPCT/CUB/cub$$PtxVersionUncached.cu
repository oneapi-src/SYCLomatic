// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

void test(int r) {
  // Start
  cub::PtxVersionUncached(r/*int*/);
  // End
}
