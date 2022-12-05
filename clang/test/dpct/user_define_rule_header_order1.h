// CHECK: #include <oneapi/dpl/execution>
// CHECK: #include <oneapi/dpl/algorithm>
// CHECK: #include <sycl/sycl.hpp>
#include <cub/cub.cuh>
#include <stddef.h>

int n, *d_in, *d_out;
void *tmp;
size_t tmp_size;

inline void test1() {
  cub::DeviceScan::InclusiveSum(tmp, tmp_size, d_in, d_out, n);
}
