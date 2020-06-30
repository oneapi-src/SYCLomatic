// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/thrust-algo.dp.cpp --match-full-lines %s
#include <vector>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/extrema.h>
#include <thrust/gather.h>
#include <thrust/binary_search.h>

void k() {
  std::vector<int> v, v2, v3, v4;

  auto up = [](int x) -> bool { return x < 23; };
  auto bp = [](int x, int y) -> bool { return x < y; };
  auto bo = [](int x, int y) -> int { return x + y; };
  auto gen = []() -> int { return 23; };

  // exclusive_scan_by_key

  // CHECK: dpstd::exclusive_scan_by_segment(dpstd::execution::seq, v.begin(), v.end(), v2.begin(), v3.begin());
  // CHECK: dpstd::exclusive_scan_by_segment(dpstd::execution::default_policy, v.begin(), v.end(), v2.begin(), v3.begin());
  thrust::exclusive_scan_by_key(thrust::seq, v.begin(), v.end(), v2.begin(), v3.begin());
  thrust::exclusive_scan_by_key(v.begin(), v.end(), v2.begin(), v3.begin());

  // CHECK: dpstd::exclusive_scan_by_segment(dpstd::execution::seq, v.begin(), v.end(), v2.begin(), v3.begin(), 1);
  // CHECK: dpstd::exclusive_scan_by_segment(dpstd::execution::default_policy, v.begin(), v.end(), v2.begin(), v3.begin(), 1);
  thrust::exclusive_scan_by_key(thrust::seq, v.begin(), v.end(), v2.begin(), v3.begin(), 1);
  thrust::exclusive_scan_by_key(v.begin(), v.end(), v2.begin(), v3.begin(), 1);

  // CHECK: dpstd::exclusive_scan_by_segment(dpstd::execution::seq, v.begin(), v.end(), v2.begin(), v3.begin(), 1, bp);
  // CHECK: dpstd::exclusive_scan_by_segment(dpstd::execution::default_policy, v.begin(), v.end(), v2.begin(), v3.begin(), 1, bp);
  thrust::exclusive_scan_by_key(thrust::seq, v.begin(), v.end(), v2.begin(), v3.begin(), 1, bp);
  thrust::exclusive_scan_by_key(v.begin(), v.end(), v2.begin(), v3.begin(), 1, bp);

  // inclusive_scan_by_key

  // CHECK: dpstd::inclusive_scan_by_segment(dpstd::execution::seq, v.begin(), v.end(), v2.begin(), v3.begin());
  // CHECK: dpstd::inclusive_scan_by_segment(dpstd::execution::default_policy, v.begin(), v.end(), v2.begin(), v3.begin());
  thrust::inclusive_scan_by_key(thrust::seq, v.begin(), v.end(), v2.begin(), v3.begin());
  thrust::inclusive_scan_by_key(v.begin(), v.end(), v2.begin(), v3.begin());

  // CHECK: dpstd::inclusive_scan_by_segment(dpstd::execution::seq, v.begin(), v.end(), v2.begin(), v3.begin(), bp);
  // CHECK: dpstd::inclusive_scan_by_segment(dpstd::execution::default_policy, v.begin(), v.end(), v2.begin(), v3.begin(), bp);
  thrust::inclusive_scan_by_key(thrust::seq, v.begin(), v.end(), v2.begin(), v3.begin(), bp);
  thrust::inclusive_scan_by_key(v.begin(), v.end(), v2.begin(), v3.begin(), bp);

  // CHECK: dpstd::inclusive_scan_by_segment(dpstd::execution::seq, v.begin(), v.end(), v2.begin(), v3.begin(), bp, bo);
  // CHECK: dpstd::inclusive_scan_by_segment(dpstd::execution::default_policy, v.begin(), v.end(), v2.begin(), v3.begin(), bp, bo);
  thrust::inclusive_scan_by_key(thrust::seq, v.begin(), v.end(), v2.begin(), v3.begin(), bp, bo);
  thrust::inclusive_scan_by_key(v.begin(), v.end(), v2.begin(), v3.begin(), bp, bo);

  // partition_point

  // CHECK: dpct::partition_point(dpstd::execution::seq, v.begin(), v.end(), up);
  thrust::partition_point(thrust::seq, v.begin(), v.end(), up);
  // CHECK: dpct::partition_point(dpstd::execution::default_policy, v.begin(), v.end(), up);
  thrust::partition_point(v.begin(), v.end(), up);


  // binary_search

  // CHECK: dpstd::binary_search(dpstd::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
  thrust::binary_search(thrust::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
  // CHECK: dpstd::binary_search(dpstd::execution::default_policy, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
  thrust::binary_search(v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());

  // CHECK: dpstd::binary_search(dpstd::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
  thrust::binary_search(thrust::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
  // CHECK: dpstd::binary_search(dpstd::execution::default_policy, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
  thrust::binary_search(v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);


  // lower_bound

  // CHECK: dpstd::lower_bound(dpstd::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
  thrust::lower_bound(thrust::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
  // CHECK: dpstd::lower_bound(dpstd::execution::default_policy, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
  thrust::lower_bound(v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());

  // CHECK: dpstd::lower_bound(dpstd::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
  thrust::lower_bound(thrust::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
  // CHECK: dpstd::lower_bound(dpstd::execution::default_policy, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
  thrust::lower_bound(v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);


  // upper_bound

  // CHECK: dpstd::upper_bound(dpstd::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
  thrust::upper_bound(thrust::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
  // CHECK: dpstd::upper_bound(dpstd::execution::default_policy, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
  thrust::upper_bound(v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());

  // CHECK: dpstd::upper_bound(dpstd::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
  thrust::upper_bound(thrust::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
  // CHECK: dpstd::upper_bound(dpstd::execution::default_policy, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
  thrust::upper_bound(v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);


  // count
  // CHECK: std::count(dpstd::execution::seq, v.begin(), v.end(), 23);
  thrust::count(thrust::seq, v.begin(), v.end(), 23);
  // CHECK: std::count(dpstd::execution::default_policy, v.begin(), v.end(), 23);
  thrust::count(v.begin(), v.end(), 23);


  // generate
  // CHECK: std::generate(dpstd::execution::seq, v.begin(), v.end(), gen);
  thrust::generate(thrust::seq, v.begin(), v.end(), gen);
  // CHECK: std::generate(dpstd::execution::default_policy, v.begin(), v.end(), gen);
  thrust::generate(v.begin(), v.end(), gen);


  // generate_n
  // CHECK: std::generate_n(dpstd::execution::seq, v.begin(), 23, gen);
  thrust::generate_n(thrust::seq, v.begin(), 23, gen);
  // CHECK: std::generate_n(dpstd::execution::default_policy, v.begin(), 23, gen);
  thrust::generate_n(v.begin(), 23, gen);


  // merge

  // CHECK: std::merge(dpstd::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
  thrust::merge(thrust::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
  // CHECK: std::merge(dpstd::execution::default_policy, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
  thrust::merge(v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());

  // CHECK: std::merge(dpstd::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
  thrust::merge(thrust::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
  // CHECK: std::merge(dpstd::execution::default_policy, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
  thrust::merge(v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
}
