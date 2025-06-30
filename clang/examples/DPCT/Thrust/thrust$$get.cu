#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/tuple.h>

void thrust_get() {
  // clang-format off
  // Start
  auto ret = thrust::make_tuple(3, 4);
  auto to = thrust::get<0>(ret);
  // End
  // clang-format on
}
