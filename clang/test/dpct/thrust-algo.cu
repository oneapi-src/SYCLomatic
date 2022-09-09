// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T/thrust-algo %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/thrust-algo/thrust-algo.dp.cpp --match-full-lines %s
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

#include <thrust/find.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>

void k() {
  std::vector<int> v, v2, v3, v4;

  auto up = [](int x) -> bool { return x < 23; };
  auto bp = [](int x, int y) -> bool { return x < y; };
  auto bo = [](int x, int y) -> int { return x + y; };
  auto gen = []() -> int { return 23; };

  thrust::maximum<int> binary_op;
  thrust::device_vector<int> tv, tv2, tv3, tv4;

  // exclusive_scan

  // CHECK: std::exclusive_scan(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), 0);
  // CHECK: std::exclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), tv.begin(), tv.end(), tv2.begin(), 0);
  thrust::exclusive_scan(thrust::host, v.begin(), v.end(), v2.begin());
  thrust::exclusive_scan(thrust::device, tv.begin(), tv.end(), tv2.begin());

  // CHECK: std::exclusive_scan(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), 0);
  // CHECK: std::exclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), tv.begin(), tv.end(), tv2.begin(), 0);
  thrust::exclusive_scan(v.begin(), v.end(), v2.begin());
  thrust::exclusive_scan(tv.begin(), tv.end(), tv2.begin());

  // CHECK: std::exclusive_scan(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), 4);
  // CHECK: std::exclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), tv.begin(), tv.end(), tv2.begin(), 4);
  thrust::exclusive_scan(thrust::host, v.begin(), v.end(), v2.begin(), 4);
  thrust::exclusive_scan(thrust::device, tv.begin(), tv.end(), tv2.begin(), 4);

  // CHECK: std::exclusive_scan(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), 4);
  // CHECK: std::exclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), tv.begin(), tv.end(), tv2.begin(), 4);
  thrust::exclusive_scan(v.begin(), v.end(), v2.begin(), 4);
  thrust::exclusive_scan(tv.begin(), tv.end(), tv2.begin(), 4);

  // CHECK: std::exclusive_scan(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), 1, binary_op);
  // CHECK: std::exclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), tv.begin(), tv.end(), tv2.begin(), 1, binary_op);
  thrust::exclusive_scan(thrust::host, v.begin(), v.end(), v2.begin(), 1, binary_op);
  thrust::exclusive_scan(thrust::device, tv.begin(), tv.end(), tv2.begin(), 1, binary_op);

  // CHECK: std::exclusive_scan(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), 1, binary_op);
  // CHECK: std::exclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), tv.begin(), tv.end(), tv2.begin(), 1, binary_op);
  thrust::exclusive_scan(v.begin(), v.end(), v2.begin(), 1, binary_op);
  thrust::exclusive_scan(tv.begin(), tv.end(), tv2.begin(), 1, binary_op);

  // exclusive_scan_by_key

  // CHECK: oneapi::dpl::exclusive_scan_by_segment(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v3.begin());
  // CHECK: oneapi::dpl::exclusive_scan_by_segment(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v3.begin());
  thrust::exclusive_scan_by_key(thrust::seq, v.begin(), v.end(), v2.begin(), v3.begin());
  thrust::exclusive_scan_by_key(v.begin(), v.end(), v2.begin(), v3.begin());

  // CHECK: oneapi::dpl::exclusive_scan_by_segment(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v3.begin(), 1);
  // CHECK: oneapi::dpl::exclusive_scan_by_segment(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v3.begin(), 1);
  thrust::exclusive_scan_by_key(thrust::seq, v.begin(), v.end(), v2.begin(), v3.begin(), 1);
  thrust::exclusive_scan_by_key(v.begin(), v.end(), v2.begin(), v3.begin(), 1);

  // CHECK: oneapi::dpl::exclusive_scan_by_segment(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v3.begin(), 1, bp);
  // CHECK: oneapi::dpl::exclusive_scan_by_segment(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v3.begin(), 1, bp);
  thrust::exclusive_scan_by_key(thrust::seq, v.begin(), v.end(), v2.begin(), v3.begin(), 1, bp);
  thrust::exclusive_scan_by_key(v.begin(), v.end(), v2.begin(), v3.begin(), 1, bp);

  // inclusive_scan_by_key

  // CHECK: oneapi::dpl::inclusive_scan_by_segment(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v3.begin());
  // CHECK: oneapi::dpl::inclusive_scan_by_segment(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v3.begin());
  thrust::inclusive_scan_by_key(thrust::seq, v.begin(), v.end(), v2.begin(), v3.begin());
  thrust::inclusive_scan_by_key(v.begin(), v.end(), v2.begin(), v3.begin());

  // CHECK: oneapi::dpl::inclusive_scan_by_segment(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v3.begin(), bp);
  // CHECK: oneapi::dpl::inclusive_scan_by_segment(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v3.begin(), bp);
  thrust::inclusive_scan_by_key(thrust::seq, v.begin(), v.end(), v2.begin(), v3.begin(), bp);
  thrust::inclusive_scan_by_key(v.begin(), v.end(), v2.begin(), v3.begin(), bp);

  // CHECK: oneapi::dpl::inclusive_scan_by_segment(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v3.begin(), bp, bo);
  // CHECK: oneapi::dpl::inclusive_scan_by_segment(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v3.begin(), bp, bo);
  thrust::inclusive_scan_by_key(thrust::seq, v.begin(), v.end(), v2.begin(), v3.begin(), bp, bo);
  thrust::inclusive_scan_by_key(v.begin(), v.end(), v2.begin(), v3.begin(), bp, bo);

  // partition_point

  // CHECK: dpct::partition_point(oneapi::dpl::execution::seq, v.begin(), v.end(), up);
  thrust::partition_point(thrust::seq, v.begin(), v.end(), up);
  // CHECK: dpct::partition_point(oneapi::dpl::execution::make_device_policy(q_ct1), v.begin(), v.end(), up);
  thrust::partition_point(v.begin(), v.end(), up);


  // binary_search

  // CHECK: oneapi::dpl::binary_search(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
  thrust::binary_search(thrust::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
  // CHECK: oneapi::dpl::binary_search(oneapi::dpl::execution::make_device_policy(q_ct1), v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
  thrust::binary_search(v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());

  // CHECK: oneapi::dpl::binary_search(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
  thrust::binary_search(thrust::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
  // CHECK: oneapi::dpl::binary_search(oneapi::dpl::execution::make_device_policy(q_ct1), v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
  thrust::binary_search(v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);

  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of thrust::binary_search is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT: thrust::binary_search(oneapi::dpl::execution::seq, v.begin(), v.end(), 1);
  thrust::binary_search(thrust::seq, v.begin(), v.end(), 1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of thrust::binary_search is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT:   thrust::binary_search(v.begin(), v.end(), 1);
  thrust::binary_search(v.begin(), v.end(), 1);

  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of thrust::binary_search is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT:   thrust::binary_search(oneapi::dpl::execution::seq, v.begin(), v.end(), 1, bp);
  thrust::binary_search(thrust::seq, v.begin(), v.end(), 1, bp);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of thrust::binary_search is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT:   thrust::binary_search(v.begin(), v.end(), 1, bp);
  thrust::binary_search(v.begin(), v.end(), 1, bp);


  // lower_bound

  // CHECK: oneapi::dpl::lower_bound(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
  thrust::lower_bound(thrust::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
  // CHECK: oneapi::dpl::lower_bound(oneapi::dpl::execution::make_device_policy(q_ct1), v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
  thrust::lower_bound(v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());

  // CHECK: oneapi::dpl::lower_bound(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
  thrust::lower_bound(thrust::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
  // CHECK: oneapi::dpl::lower_bound(oneapi::dpl::execution::make_device_policy(q_ct1), v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
  thrust::lower_bound(v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);


  // upper_bound

  // CHECK: oneapi::dpl::upper_bound(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
  thrust::upper_bound(thrust::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
  // CHECK: oneapi::dpl::upper_bound(oneapi::dpl::execution::make_device_policy(q_ct1), v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
  thrust::upper_bound(v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());

  // CHECK: oneapi::dpl::upper_bound(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
  thrust::upper_bound(thrust::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
  // CHECK: oneapi::dpl::upper_bound(oneapi::dpl::execution::make_device_policy(q_ct1), v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
  thrust::upper_bound(v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);


  // count
  // CHECK: std::count(oneapi::dpl::execution::seq, v.begin(), v.end(), 23);
  thrust::count(thrust::seq, v.begin(), v.end(), 23);
  // CHECK: std::count(oneapi::dpl::execution::make_device_policy(q_ct1), v.begin(), v.end(), 23);
  thrust::count(v.begin(), v.end(), 23);


  // generate
  // CHECK: std::generate(oneapi::dpl::execution::seq, v.begin(), v.end(), gen);
  thrust::generate(thrust::seq, v.begin(), v.end(), gen);
  // CHECK: std::generate(oneapi::dpl::execution::make_device_policy(q_ct1), v.begin(), v.end(), gen);
  thrust::generate(v.begin(), v.end(), gen);


  // generate_n
  // CHECK: std::generate_n(oneapi::dpl::execution::seq, v.begin(), 23, gen);
  thrust::generate_n(thrust::seq, v.begin(), 23, gen);
  // CHECK: std::generate_n(oneapi::dpl::execution::make_device_policy(q_ct1), v.begin(), 23, gen);
  thrust::generate_n(v.begin(), 23, gen);


  // merge

  // CHECK: std::merge(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
  thrust::merge(thrust::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
  // CHECK: std::merge(oneapi::dpl::execution::make_device_policy(q_ct1), v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
  thrust::merge(v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());

  // CHECK: std::merge(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
  thrust::merge(thrust::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
  // CHECK: std::merge(oneapi::dpl::execution::make_device_policy(q_ct1), v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
  thrust::merge(v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
}


void foo(cudaStream_t stream) {
  //CHECK:std::vector<int> h;
  //CHECK-NEXT:dpct::device_vector<int> d;
  thrust::host_vector<int> h;
  thrust::device_vector<int> d;

  //thrust::find
  //CHECK:oneapi::dpl::find(oneapi::dpl::execution::seq, h.begin(), h.end(), 1);
  //CHECK-NEXT:oneapi::dpl::find(oneapi::dpl::execution::seq, h.begin(), h.end(), 1);
  //CHECK-NEXT:oneapi::dpl::find(oneapi::dpl::execution::make_device_policy(q_ct1), d.begin(), d.end(), 1);
  thrust::find(thrust::seq, h.begin(), h.end(), 1);
  thrust::find(h.begin(), h.end(), 1);
  thrust::find(d.begin(), d.end(), 1);

  //thrust::sort_by_key
  //CHECK:dpct::sort(oneapi::dpl::execution::seq, h.begin(), h.end(), h.begin(), std::greater<int>());
  //CHECK-NEXT:dpct::sort(oneapi::dpl::execution::seq, h.begin(), h.end(), h.begin());
  //CHECK-NEXT:dpct::sort(oneapi::dpl::execution::make_device_policy(q_ct1), d.begin(), d.end(), h.begin());
  //CHECK-NEXT:dpct::sort(oneapi::dpl::execution::make_device_policy(q_ct1), d.begin(), d.end(), d.begin());
  //CHECK-NEXT:dpct::sort(oneapi::dpl::execution::seq, h.begin(), h.end(), h.begin(), std::greater<int>());
  //CHECK-NEXT:dpct::sort(oneapi::dpl::execution::make_device_policy(q_ct1), d.begin(), d.end(), d.begin(), std::greater<int>());
  thrust::sort_by_key(thrust::seq, h.begin(), h.end(), h.begin(), thrust::greater<int>());
  thrust::sort_by_key(h.begin(), h.end(), h.begin());
  thrust::sort_by_key(d.begin(), d.end(), h.begin());
  thrust::sort_by_key(thrust::device, d.begin(), d.end(), d.begin());
  thrust::sort_by_key(h.begin(), h.end(), h.begin(), thrust::greater<int>());
  thrust::sort_by_key(d.begin(), d.end(), d.begin(), thrust::greater<int>());

  //CHECK:std::multiplies<int> bo1;
  //CHECK-NEXT:std::multiplies<int> bo2;
  thrust::multiplies<int> bo1;
  thrust::multiplies<int> bo2;
  //thrust::inner_product
  //CHECK:dpct::inner_product(oneapi::dpl::execution::seq, h.begin(), h.end(), h.begin(), 1);
  //CHECK-NEXT:dpct::inner_product(oneapi::dpl::execution::make_device_policy(q_ct1), d.begin(), d.end(), d.begin(), 1, bo1, bo2);
  //CHECK-NEXT:dpct::inner_product(oneapi::dpl::execution::seq, h.begin(), h.end(), h.begin(), 1);
  //CHECK-NEXT:dpct::inner_product(oneapi::dpl::execution::make_device_policy(q_ct1), d.begin(), d.end(), d.begin(), 1);
  //CHECK-NEXT:dpct::inner_product(oneapi::dpl::execution::seq, h.begin(), h.end(), h.begin(), 1, bo1, bo2);
  //CHECK-NEXT:dpct::inner_product(oneapi::dpl::execution::make_device_policy(q_ct1), d.begin(), d.end(), d.begin(), 1, bo1, bo2);
  thrust::inner_product(thrust::host, h.begin(), h.end(), h.begin(), 1);
  thrust::inner_product(thrust::device, d.begin(), d.end(), d.begin(), 1, bo1, bo2);
  thrust::inner_product(h.begin(), h.end(), h.begin(), 1);
  thrust::inner_product(d.begin(), d.end(), d.begin(), 1);
  thrust::inner_product(h.begin(), h.end(), h.begin(), 1, bo1, bo2);
  thrust::inner_product(d.begin(), d.end(), d.begin(), 1, bo1, bo2);

  //CHECK:std::not_equal_to<int> bp;
  thrust::not_equal_to<int> bp;
  //thrust::reduce_by_key
  //CHECK:oneapi::dpl::reduce_by_segment(oneapi::dpl::execution::seq, h.begin(), h.end(), h.begin(), h.end(), h.begin(), bp, bo1);
  //CHECK-NEXT:oneapi::dpl::reduce_by_segment(oneapi::dpl::execution::make_device_policy(q_ct1), d.begin(), d.end(), d.begin(), d.end(), d.begin(), bp);
  //CHECK-NEXT:oneapi::dpl::reduce_by_segment(oneapi::dpl::execution::seq, h.begin(), h.end(), h.begin(), h.end(), h.begin(), bp, bo1);
  //CHECK-NEXT:oneapi::dpl::reduce_by_segment(oneapi::dpl::execution::make_device_policy(q_ct1), d.begin(), d.end(), d.begin(), d.end(), d.begin(), bp, bo1);
  //CHECK-NEXT:oneapi::dpl::reduce_by_segment(oneapi::dpl::execution::seq, h.begin(), h.end(), dpct::constant_iterator<int>(1), h.end(), h.begin());
  //CHECK-NEXT:oneapi::dpl::reduce_by_segment(oneapi::dpl::execution::seq, h.begin(), h.end(), h.begin(), h.end(), h.begin(), bp);
  //CHECK-NEXT:oneapi::dpl::reduce_by_segment(oneapi::dpl::execution::make_device_policy(q_ct1), d.begin(), d.end(), d.begin(), d.end(), d.begin(), bp);
  //CHECK-NEXT:oneapi::dpl::reduce_by_segment(oneapi::dpl::execution::seq, h.begin(), h.end(), h.begin(), h.end(), h.begin());
  //CHECK-NEXT:oneapi::dpl::reduce_by_segment(oneapi::dpl::execution::make_device_policy(q_ct1), d.begin(), d.end(), d.begin(), d.end(), d.begin());
  thrust::reduce_by_key(thrust::host, h.begin(), h.end(), h.begin(), h.end(), h.begin(), bp, bo1);
  thrust::reduce_by_key(thrust::device, d.begin(), d.end(), d.begin(), d.end(), d.begin(), bp);
  thrust::reduce_by_key(h.begin(), h.end(), h.begin(), h.end(), h.begin(), bp, bo1);
  thrust::reduce_by_key(d.begin(), d.end(), d.begin(), d.end(), d.begin(), bp, bo1);
  thrust::reduce_by_key(thrust::host, h.begin(), h.end(), thrust::constant_iterator<int>(1), h.end(), h.begin());
  thrust::reduce_by_key(h.begin(), h.end(), h.begin(), h.end(), h.begin(), bp);
  thrust::reduce_by_key(d.begin(), d.end(), d.begin(), d.end(), d.begin(), bp);
  thrust::reduce_by_key(h.begin(), h.end(), h.begin(), h.end(), h.begin());
  thrust::reduce_by_key(d.begin(), d.end(), d.begin(), d.end(), d.begin());

  {
    //CHECK:std::vector<int> h_keys,h_values;
    //CHECK-NEXT:dpct::device_vector<int> d_keys, d_values;
    //CHECK-NEXT:oneapi::dpl::equal_to<int> binary_pred;
    thrust::host_vector<int> h_keys,h_values;
    thrust::device_vector<int> d_keys, d_values;
    thrust::equal_to<int> binary_pred;

    //CHECK:dpct::unique(oneapi::dpl::execution::seq, h_keys.begin(), h_keys.end(), h_values.begin());
    //CHECK-NEXT:dpct::unique(oneapi::dpl::execution::seq, h_keys.begin(), h_keys.end(), h_values.begin());
    //CHECK-NEXT:dpct::unique(oneapi::dpl::execution::seq, h_keys.begin(), h_keys.end(), h_values.begin(), binary_pred);
    //CHECK-NEXT:dpct::unique(oneapi::dpl::execution::seq, h_keys.begin(), h_keys.end(), h_values.begin(), binary_pred);
    //CHECK-NEXT:dpct::unique(oneapi::dpl::execution::make_device_policy(q_ct1), d_keys.begin(), d_keys.end(), d_values.begin());
    //CHECK-NEXT:dpct::unique(oneapi::dpl::execution::make_device_policy(q_ct1), d_keys.begin(), d_keys.end(), d_values.begin());
    //CHECK-NEXT:dpct::unique(oneapi::dpl::execution::make_device_policy(q_ct1), d_keys.begin(), d_keys.end(), d_values.begin(), binary_pred);
    //CHECK-NEXT:dpct::unique(oneapi::dpl::execution::make_device_policy(q_ct1), d_keys.begin(), d_keys.end(), d_values.begin(), binary_pred);
    thrust::unique_by_key(thrust::host, h_keys.begin(), h_keys.end(), h_values.begin());
    thrust::unique_by_key(h_keys.begin(), h_keys.end(), h_values.begin());
    thrust::unique_by_key(thrust::host, h_keys.begin(), h_keys.end(),h_values.begin(), binary_pred);
    thrust::unique_by_key(h_keys.begin(), h_keys.end(),h_values.begin(), binary_pred);
    thrust::unique_by_key(thrust::device, d_keys.begin(), d_keys.end(), d_values.begin());
    thrust::unique_by_key(d_keys.begin(), d_keys.end(), d_values.begin());
    thrust::unique_by_key(thrust::device, d_keys.begin(), d_keys.end(), d_values.begin(), binary_pred);
    thrust::unique_by_key(d_keys.begin(), d_keys.end(), d_values.begin(), binary_pred);
  }
}
