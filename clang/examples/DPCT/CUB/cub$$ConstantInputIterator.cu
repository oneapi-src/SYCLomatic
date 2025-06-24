// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

void test(int d_in) {
  // Start
  cub::ConstantInputIterator<int> Iter(d_in);
  // End
}
