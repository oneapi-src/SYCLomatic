
#include <thrust/tuple.h>

void thrust_proj() {
  // clang-format off
  // Start
  double a, b;
  thrust::tie(a, b) = thrust::make_tuple(1.0, 2.0);
  // End
  // clang-format on
}
