#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/tuple.h>
#include <thrust/device_ptr.h>
#include <thrust/count.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

void egenerate_n_test() {
  // clang-format off
  // Start
  std::vector<int> v, v2, v3, v4;
  auto gen = []() -> int { return 23; };
  /*1*/ thrust::generate_n(thrust::seq, v.begin(), 23, gen);
  /*2*/ thrust::generate_n(v.begin(), 23, gen);
  // End
  // clang-format on
}
