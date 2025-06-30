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

void generate_test() {
  // clang-format off
  // Start
  std::vector<int> v, v2, v3, v4;
  auto gen = []() -> int { return 23; };
  /*1*/ thrust::generate(thrust::seq, v.begin(), v.end(), gen);
  /*2*/ thrust::generate(v.begin(), v.end(), gen);
  // End
  // clang-format on
}