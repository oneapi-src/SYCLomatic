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

void exclusive_scan_by_key_test() {
  // clang-format off
  // Start
  std::vector<int> v, v2, v3, v4;
  auto bp = [](int x, int y) -> bool { return x < y; };
  thrust::device_vector<int> tv, tv2, tv3, tv4;
  /*1*/ thrust::exclusive_scan_by_key(thrust::seq, v.begin(), v.end(), v2.begin(), v3.begin());
  /*2*/ thrust::exclusive_scan_by_key(v.begin(), v.end(), v2.begin(), v3.begin());
  /*3*/ thrust::exclusive_scan_by_key(thrust::seq, v.begin(), v.end(), v2.begin(), v3.begin(), 1);
  /*4*/ thrust::exclusive_scan_by_key(v.begin(), v.end(), v2.begin(), v3.begin(), 1);
  /*5*/ thrust::exclusive_scan_by_key(thrust::seq, v.begin(), v.end(), v2.begin(), v3.begin(), 1, bp);
  /*6*/ thrust::exclusive_scan_by_key(v.begin(), v.end(), v2.begin(), v3.begin(), 1, bp);
  // End
  // clang-format on
}
