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
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/partition.h>
#include <thrust/scatter.h>

void sort_by_key() {
  // clang-format off
  // Start
  thrust::host_vector<int> h;
  thrust::device_vector<int> d;
  /*1*/ thrust::sort_by_key(thrust::seq, h.begin(), h.end(), h.begin(), thrust::greater<int>());
  /*2*/ thrust::sort_by_key(h.begin(), h.end(), h.begin());
  /*3*/ thrust::sort_by_key(thrust::device, d.begin(), d.end(), d.begin());
  /*4*/ thrust::sort_by_key(h.begin(), h.end(), h.begin(), thrust::greater<int>());
  // End
  // clang-format on
}
