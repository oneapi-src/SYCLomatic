#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/find.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>
#include <thrust/mismatch.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/replace.h>
#include <thrust/reverse.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/transform_scan.h>
#include <thrust/unique.h>

// CHECK: #include <oneapi/dpl/memory>
#include <thrust/equal.h>
#include <thrust/uninitialized_copy.h>

// for cuda 12.0
#include <thrust/iterator/constant_iterator.h>
#include <thrust/partition.h>
#include <thrust/scatter.h>

void merge_test() {
  // clang-format off
  // Start
  std::vector<int> v, v2, v3, v4;
  auto bp = [](int x, int y) -> bool { return x < y; };
  thrust::device_vector<int> tv, tv2, tv3, tv4;
  /*1*/ thrust::merge(thrust::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
  /*2*/ thrust::merge(v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
  /*3*/ thrust::merge(thrust::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
  /*4*/ thrust::merge(v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
  // End
  // clang-format on
}
