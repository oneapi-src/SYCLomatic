#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <thrust/adjacent_difference.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>

void inclusive_scan_by_key_test() {
  // clang-format off
  // Start
  std::vector<int> v, v2, v3, v4;
  auto bp = [](int x, int y) -> bool { return x < y; };
  auto bo = [](int x, int y) -> int { return x + y; };
  thrust::device_vector<int> tv, tv2, tv3, tv4;
  /*1*/ thrust::inclusive_scan_by_key(v.begin(), v.end(), v2.begin(), v3.begin());
  /*2*/ thrust::inclusive_scan_by_key(thrust::seq, v.begin(), v.end(), v2.begin(), v3.begin(), bp);
  /*3*/ thrust::inclusive_scan_by_key(v.begin(), v.end(), v2.begin(), v3.begin(), bp);
  /*4*/ thrust::inclusive_scan_by_key(v.begin(), v.end(), v2.begin(), v3.begin(), bp, bo);
  // End
  // clang-format on
}
