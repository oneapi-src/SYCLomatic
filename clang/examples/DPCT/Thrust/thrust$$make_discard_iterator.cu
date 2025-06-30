#include <thrust/scan.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>

void malloc_make_discard_iterator() {
  // clang-format off
  // Start
  thrust::make_discard_iterator();
  // End
  // clang-format on
}
