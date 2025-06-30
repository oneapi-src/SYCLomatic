#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/unique.h>

void malloc_make_counting_iterator() {
  // clang-format off
  // Start
  auto range = thrust::make_counting_iterator(0);
  // End
  // clang-format on
}
