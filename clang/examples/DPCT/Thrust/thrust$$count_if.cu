#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>
#include <thrust/scatter.h>
#include <thrust/tuple.h>

void count_if_test() {

  // clang-format off
  // Start
  std::vector<thrust::device_vector<int>> d(10);
  auto t = thrust::make_counting_iterator(0);
  int ret = thrust::count_if(t, t + 10, [=] __device__(int idx) { return true;});
  // End
  // clang-format on
}
