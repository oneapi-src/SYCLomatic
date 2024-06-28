#include <algorithm>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/scatter.h>
#include <thrust/tuple.h>
// for cuda 12.0
#include <thrust/count.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

void raw_pointer_cast_test(void) {
  std::vector<thrust::device_vector<int>> d(10);

  // Start
  auto min_costs_ptr = thrust::raw_pointer_cast(d[0].data());
  // End
}
