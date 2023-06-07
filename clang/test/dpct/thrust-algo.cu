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
#include <thrust/unique.h>
#include <thrust/find.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/transform_scan.h>
#include <thrust/set_operations.h>
#include <thrust/tabulate.h>
#include <thrust/functional.h>
#include <thrust/remove.h>
#include <thrust/mismatch.h>
#include <thrust/replace.h>
#include <thrust/reverse.h>

// for cuda 12.0
#include <thrust/iterator/constant_iterator.h>
#include <thrust/partition.h>


void k() {
  std::vector<int> v, v2, v3, v4;

  auto up = [](int x) -> bool { return x < 23; };
  auto bp = [](int x, int y) -> bool { return x < y; };
  auto bo = [](int x, int y) -> int { return x + y; };
  auto gen = []() -> int { return 23; };

  thrust::maximum<int> binary_op;
  thrust::device_vector<int> tv, tv2, tv3, tv4;

  // exclusive_scan

  // CHECK: std::exclusive_scan(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), (decltype(v2.begin())::value_type)0);
  // CHECK: std::exclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), tv.begin(), tv.end(), tv2.begin(), (decltype(tv2.begin())::value_type)0);
  thrust::exclusive_scan(thrust::host, v.begin(), v.end(), v2.begin());
  thrust::exclusive_scan(thrust::device, tv.begin(), tv.end(), tv2.begin());

  // CHECK: std::exclusive_scan(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), (decltype(v2.begin())::value_type)0);
  // CHECK: std::exclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), tv.begin(), tv.end(), tv2.begin(), (decltype(tv2.begin())::value_type)0);
  thrust::exclusive_scan(v.begin(), v.end(), v2.begin());
  thrust::exclusive_scan(tv.begin(), tv.end(), tv2.begin());

  // CHECK: std::exclusive_scan(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), (decltype(v2.begin())::value_type)4);
  // CHECK: std::exclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), tv.begin(), tv.end(), tv2.begin(), (decltype(tv2.begin())::value_type)4);
  thrust::exclusive_scan(thrust::host, v.begin(), v.end(), v2.begin(), 4);
  thrust::exclusive_scan(thrust::device, tv.begin(), tv.end(), tv2.begin(), 4);

  // CHECK: std::exclusive_scan(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), (decltype(v2.begin())::value_type)4);
  // CHECK: std::exclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), tv.begin(), tv.end(), tv2.begin(), (decltype(tv2.begin())::value_type)4);
  thrust::exclusive_scan(v.begin(), v.end(), v2.begin(), 4);
  thrust::exclusive_scan(tv.begin(), tv.end(), tv2.begin(), 4);

  // CHECK: std::exclusive_scan(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), (decltype(v2.begin())::value_type)1, binary_op);
  // CHECK: std::exclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), tv.begin(), tv.end(), tv2.begin(), (decltype(tv2.begin())::value_type)1, binary_op);
  thrust::exclusive_scan(thrust::host, v.begin(), v.end(), v2.begin(), 1, binary_op);
  thrust::exclusive_scan(thrust::device, tv.begin(), tv.end(), tv2.begin(), 1, binary_op);

  // CHECK: std::exclusive_scan(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), (decltype(v2.begin())::value_type)1, binary_op);
  // CHECK: std::exclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), tv.begin(), tv.end(), tv2.begin(), (decltype(tv2.begin())::value_type)1, binary_op);
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
  // CHECK: dpct::partition_point(oneapi::dpl::execution::seq, v.begin(), v.end(), up);
  thrust::partition_point(v.begin(), v.end(), up);


  // binary_search

  // CHECK: oneapi::dpl::binary_search(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
  thrust::binary_search(thrust::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
  // CHECK: oneapi::dpl::binary_search(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
  thrust::binary_search(v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());

  // CHECK: oneapi::dpl::binary_search(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
  thrust::binary_search(thrust::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
  // CHECK: oneapi::dpl::binary_search(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
  thrust::binary_search(v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);

  // CHECK: oneapi::dpl::binary_search(oneapi::dpl::execution::seq, v.begin(), v.end(), 1);
  thrust::binary_search(thrust::seq, v.begin(), v.end(), 1);

  // CHECK:  oneapi::dpl::binary_search(oneapi::dpl::execution::seq, v.begin(), v.end(), 1);
  thrust::binary_search(v.begin(), v.end(), 1);

  // CHECK: oneapi::dpl::binary_search(oneapi::dpl::execution::seq, v.begin(), v.end(), 1, bp);
  thrust::binary_search(thrust::seq, v.begin(), v.end(), 1, bp);

  // CHECK: oneapi::dpl::binary_search(oneapi::dpl::execution::seq, v.begin(), v.end(), 1, bp);
  thrust::binary_search(v.begin(), v.end(), 1, bp);


  // lower_bound

  // CHECK: oneapi::dpl::lower_bound(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
  thrust::lower_bound(thrust::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
  // CHECK: oneapi::dpl::lower_bound(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
  thrust::lower_bound(v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());

  // CHECK: oneapi::dpl::lower_bound(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
  thrust::lower_bound(thrust::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
  // CHECK: oneapi::dpl::lower_bound(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
  thrust::lower_bound(v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);


  // upper_bound

  // CHECK: oneapi::dpl::upper_bound(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
  thrust::upper_bound(thrust::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
  // CHECK: oneapi::dpl::upper_bound(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
  thrust::upper_bound(v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());

  // CHECK: oneapi::dpl::upper_bound(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
  thrust::upper_bound(thrust::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
  // CHECK: oneapi::dpl::upper_bound(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
  thrust::upper_bound(v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);


  // count
  // CHECK: std::count(oneapi::dpl::execution::seq, v.begin(), v.end(), 23);
  thrust::count(thrust::seq, v.begin(), v.end(), 23);
  // CHECK: std::count(oneapi::dpl::execution::seq, v.begin(), v.end(), 23);
  thrust::count(v.begin(), v.end(), 23);


  // generate
  // CHECK: std::generate(oneapi::dpl::execution::seq, v.begin(), v.end(), gen);
  thrust::generate(thrust::seq, v.begin(), v.end(), gen);
  // CHECK: std::generate(oneapi::dpl::execution::seq, v.begin(), v.end(), gen);
  thrust::generate(v.begin(), v.end(), gen);


  // generate_n
  // CHECK: std::generate_n(oneapi::dpl::execution::seq, v.begin(), 23, gen);
  thrust::generate_n(thrust::seq, v.begin(), 23, gen);
  // CHECK: std::generate_n(oneapi::dpl::execution::seq, v.begin(), 23, gen);
  thrust::generate_n(v.begin(), 23, gen);


  // merge

  // CHECK: std::merge(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
  thrust::merge(thrust::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
  // CHECK: std::merge(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
  thrust::merge(v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());

  // CHECK: std::merge(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
  thrust::merge(thrust::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
  // CHECK: std::merge(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
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
// CHECK:std::vector<int> h_keys,h_values;
// CHECK-NEXT:dpct::device_vector<int> d_keys, d_values;
// CHECK-NEXT:oneapi::dpl::equal_to<int> binary_pred;
    thrust::host_vector<int> h_keys,h_values;
    thrust::device_vector<int> d_keys, d_values;
    thrust::equal_to<int> binary_pred;
    const int N = 7;
    int A[N]; // keys
    int B[N]; // values

// CHECK:dpct::unique(oneapi::dpl::execution::seq, h_keys.begin(), h_keys.end(), h_values.begin());
// CHECK-NEXT:dpct::unique(oneapi::dpl::execution::seq, h_keys.begin(), h_keys.end(), h_values.begin());
// CHECK-NEXT:dpct::unique(oneapi::dpl::execution::seq, h_keys.begin(), h_keys.end(), h_values.begin(), binary_pred);
// CHECK-NEXT:dpct::unique(oneapi::dpl::execution::seq, h_keys.begin(), h_keys.end(), h_values.begin(), binary_pred);
// CHECK-NEXT:dpct::unique(oneapi::dpl::execution::make_device_policy(q_ct1), d_keys.begin(), d_keys.end(), d_values.begin());
// CHECK-NEXT:dpct::unique(oneapi::dpl::execution::make_device_policy(q_ct1), d_keys.begin(), d_keys.end(), d_values.begin());
// CHECK-NEXT:dpct::unique(oneapi::dpl::execution::make_device_policy(q_ct1), d_keys.begin(), d_keys.end(), d_values.begin(), binary_pred);
// CHECK-NEXT:dpct::unique(oneapi::dpl::execution::make_device_policy(q_ct1), d_keys.begin(), d_keys.end(), d_values.begin(), binary_pred);
// CHECK-NEXT:dpct::unique(oneapi::dpl::execution::seq, A, A + N, B);
// CHECK-NEXT:dpct::unique(oneapi::dpl::execution::seq, A, A + N, B);
// CHECK-NEXT:dpct::unique(oneapi::dpl::execution::seq, A, A + N, B, binary_pred);
// CHECK-NEXT:dpct::unique(oneapi::dpl::execution::seq, A, A + N, B, binary_pred);
    thrust::unique_by_key(thrust::host, h_keys.begin(), h_keys.end(), h_values.begin());
    thrust::unique_by_key(h_keys.begin(), h_keys.end(), h_values.begin());
    thrust::unique_by_key(thrust::host, h_keys.begin(), h_keys.end(),h_values.begin(), binary_pred);
    thrust::unique_by_key(h_keys.begin(), h_keys.end(),h_values.begin(), binary_pred);
    thrust::unique_by_key(thrust::device, d_keys.begin(), d_keys.end(), d_values.begin());
    thrust::unique_by_key(d_keys.begin(), d_keys.end(), d_values.begin());
    thrust::unique_by_key(thrust::device, d_keys.begin(), d_keys.end(), d_values.begin(), binary_pred);
    thrust::unique_by_key(d_keys.begin(), d_keys.end(), d_values.begin(), binary_pred);
    thrust::unique_by_key(thrust::host, A, A + N, B);
    thrust::unique_by_key(A, A + N, B);
    thrust::unique_by_key(thrust::host, A, A + N, B, binary_pred);
    thrust::unique_by_key(A, A + N, B, binary_pred);

  }
}


struct key_value
	{
		int key;
		int value;
		__host__ __device__ 
		bool operator!=( struct key_value &tmp) const {
			if (this->key != tmp.key||this->value != tmp.value) {
				return true;
			}
			else {
				return false;
			}

		}
	};

struct compare_key_value
	{
		__host__ __device__
			bool operator()(int lhs, int rhs) const {
			return lhs < rhs;
		}
	};

void minmax_element_test() {
	const int N = 6;
	int data[N] = { 1, 0, 2, 2, 1, 3 };
	thrust::host_vector<int> h_values(data, data + N);
	thrust::device_vector<int> d_values(data, data + N);

// CHECK:  oneapi::dpl::minmax_element(oneapi::dpl::execution::seq, h_values.begin(), h_values.end());
// CHECK-NEXT:	oneapi::dpl::minmax_element(oneapi::dpl::execution::seq, h_values.begin(), h_values.end());
// CHECK-NEXT:  oneapi::dpl::minmax_element(oneapi::dpl::execution::seq, h_values.begin(), h_values.begin() + 4, compare_key_value());
// CHECK-NEXT:  oneapi::dpl::minmax_element(oneapi::dpl::execution::seq, h_values.begin(), h_values.begin() + 4, compare_key_value());
// CHECK-NEXT:  oneapi::dpl::minmax_element(oneapi::dpl::execution::make_device_policy(q_ct1), d_values.begin(), d_values.end());
// CHECK-NEXT:  oneapi::dpl::minmax_element(oneapi::dpl::execution::make_device_policy(q_ct1), d_values.begin(), d_values.end());
// CHECK-NEXT:  oneapi::dpl::minmax_element(oneapi::dpl::execution::make_device_policy(q_ct1), d_values.begin(), d_values.end(), compare_key_value());
// CHECK-NEXT:  oneapi::dpl::minmax_element(oneapi::dpl::execution::make_device_policy(q_ct1), d_values.begin(), d_values.end(), compare_key_value());
// CHECK-NEXT:  oneapi::dpl::minmax_element(oneapi::dpl::execution::seq, data, data+N);
// CHECK-NEXT:  oneapi::dpl::minmax_element(oneapi::dpl::execution::seq, data, data+N);
// CHECK-NEXT:  oneapi::dpl::minmax_element(oneapi::dpl::execution::seq, data, data+N, compare_key_value());
// CHECK-NEXT:  oneapi::dpl::minmax_element(oneapi::dpl::execution::seq, data, data+N, compare_key_value());
  thrust::minmax_element(thrust::host, h_values.begin(), h_values.end());
	thrust::minmax_element(h_values.begin(), h_values.end());
  thrust::minmax_element(thrust::host, h_values.begin(), h_values.begin() + 4, compare_key_value());
  thrust::minmax_element(h_values.begin(), h_values.begin() + 4, compare_key_value());
  thrust::minmax_element(thrust::device, d_values.begin(), d_values.end());
  thrust::minmax_element(d_values.begin(), d_values.end());
  thrust::minmax_element(thrust::device, d_values.begin(), d_values.end(), compare_key_value());
  thrust::minmax_element(d_values.begin(), d_values.end(), compare_key_value());
  thrust::minmax_element(thrust::host, data, data+N);
  thrust::minmax_element(data, data+N);
  thrust::minmax_element(thrust::host, data, data+N, compare_key_value());
  thrust::minmax_element(data, data+N, compare_key_value());
}

void is_sorted_test() {
    const int N=6;
    int datas[N]={1,4,2,8,5,7};

// CHECK:    std::vector<int> h_v(datas,datas+N);
// CHECK-NEXT:    dpct::device_vector<int> d_v(datas,datas+N);
// CHECK-NEXT:    std::greater<int> comp;
    thrust::host_vector<int> h_v(datas,datas+N);
    thrust::device_vector<int> d_v(datas,datas+N);
    thrust::greater<int> comp;

// CHECK:    oneapi::dpl::is_sorted(oneapi::dpl::execution::seq, h_v.begin(), h_v.end());
// CHECK-NEXT:    oneapi::dpl::is_sorted(oneapi::dpl::execution::seq, h_v.begin(), h_v.end());
// CHECK-NEXT:    oneapi::dpl::is_sorted(oneapi::dpl::execution::seq, h_v.begin(), h_v.end(), comp);
// CHECK-NEXT:    oneapi::dpl::is_sorted(oneapi::dpl::execution::seq, h_v.begin(), h_v.end(), comp);
// CHECK-NEXT:    oneapi::dpl::is_sorted(oneapi::dpl::execution::make_device_policy(q_ct1), d_v.begin(), d_v.end());
// CHECK-NEXT:    oneapi::dpl::is_sorted(oneapi::dpl::execution::make_device_policy(q_ct1), d_v.begin(), d_v.end(), comp);
// CHECK-NEXT:    oneapi::dpl::is_sorted(oneapi::dpl::execution::make_device_policy(q_ct1), d_v.begin(), d_v.end(), comp);
// CHECK-NEXT:    oneapi::dpl::is_sorted(oneapi::dpl::execution::seq, datas, datas+N);
// CHECK-NEXT:    oneapi::dpl::is_sorted(oneapi::dpl::execution::seq, datas, datas+N);
// CHECK-NEXT:    oneapi::dpl::is_sorted(oneapi::dpl::execution::seq, datas, datas+N, comp);
// CHECK-NEXT:    oneapi::dpl::is_sorted(oneapi::dpl::execution::seq, datas, datas+N, comp);
  thrust::is_sorted(thrust::host, h_v.begin(), h_v.end());
  thrust::is_sorted( h_v.begin(), h_v.end());
  thrust::is_sorted(thrust::host, h_v.begin(), h_v.end(),comp);
  thrust::is_sorted( h_v.begin(), h_v.end(),comp);
  thrust::is_sorted(thrust::device, d_v.begin(), d_v.end());
  thrust::is_sorted(thrust::device, d_v.begin(), d_v.end(),comp);
  thrust::is_sorted( d_v.begin(), d_v.end(),comp);
  thrust::is_sorted(thrust::host, datas, datas+N);
  thrust::is_sorted( datas, datas+N);
  thrust::is_sorted(thrust::host,datas, datas+N,comp);
  thrust::is_sorted(datas, datas+N,comp);
}

struct is_even
{
  __host__ __device__
  bool operator()(const int &x)
  {
    return (x % 2) == 0;
  }
};
void is_partition_test() {
  int datas[]={1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int ans[]={2, 4, 6, 8, 10, 1, 3, 5, 7, 9};
  const int N=sizeof(datas)/sizeof(int);
  int stencil[N]={1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  thrust::host_vector<int> h_vdata(datas,datas+N);
  thrust::host_vector<int> h_vstencil(stencil,stencil+N);
  thrust::device_vector<int> d_v(datas,datas+N);
  thrust::host_vector<int> h_v(datas,datas+N);
  thrust::device_vector<int> d_vdata(datas,datas+N);
  thrust::device_vector<int> d_vstencil(stencil,stencil+N);

// CHECK:  oneapi::dpl::partition(oneapi::dpl::execution::seq, h_v.begin(), h_v.end(), is_even());
// CHECK-NEXT:  oneapi::dpl::partition(oneapi::dpl::execution::seq, h_v.begin(), h_v.end(), is_even());
// CHECK-NEXT:  dpct::partition(oneapi::dpl::execution::seq, h_vdata.begin(), h_vdata.end(), h_vstencil.begin(), is_even());
// CHECK-NEXT:  dpct::partition(oneapi::dpl::execution::seq, h_vdata.begin(), h_vdata.end(), h_vstencil.begin(), is_even());
// CHECK-NEXT:  oneapi::dpl::partition(oneapi::dpl::execution::make_device_policy(q_ct1), d_v.begin(), d_v.end(), is_even());
// CHECK-NEXT:  oneapi::dpl::partition(oneapi::dpl::execution::make_device_policy(q_ct1), d_v.begin(), d_v.end(), is_even());
// CHECK-NEXT:  dpct::partition(oneapi::dpl::execution::make_device_policy(q_ct1), d_vdata.begin(), d_vdata.end(), d_vstencil.begin(), is_even());
// CHECK-NEXT:  dpct::partition(oneapi::dpl::execution::make_device_policy(q_ct1), d_vdata.begin(), d_vdata.end(), d_vstencil.begin(), is_even());
// CHECK-NEXT:  oneapi::dpl::partition(oneapi::dpl::execution::seq, datas, datas+N, is_even());
// CHECK-NEXT:  oneapi::dpl::partition(oneapi::dpl::execution::seq, datas, datas+N, is_even());
// CHECK-NEXT:  dpct::partition(oneapi::dpl::execution::seq, datas, datas+N, stencil, is_even());
// CHECK-NEXT:  dpct::partition(oneapi::dpl::execution::seq, datas, datas+N, stencil, is_even());
  thrust::partition(thrust::host, h_v.begin(), h_v.end(),is_even());
  thrust::partition( h_v.begin(), h_v.end(),is_even());
  thrust::partition(thrust::host, h_vdata.begin(), h_vdata.end(),h_vstencil.begin(),is_even());
  thrust::partition(h_vdata.begin(), h_vdata.end(),h_vstencil.begin(),is_even());
  thrust::partition(thrust::device, d_v.begin(), d_v.end(),is_even());
  thrust::partition( d_v.begin(), d_v.end(),is_even());
  thrust::partition(thrust::device, d_vdata.begin(), d_vdata.end(),d_vstencil.begin(),is_even());
  thrust::partition( d_vdata.begin(), d_vdata.end(),d_vstencil.begin(),is_even());
  thrust::partition(thrust::host, datas, datas+N,is_even());
  thrust::partition( datas, datas+N,is_even());
  thrust::partition(thrust::host,  datas, datas+N, stencil,is_even());
  thrust::partition( datas, datas+N, stencil,is_even());
}


void unique_copy_test() {
  const int N=7;
  int A[N]={1, 3, 3, 3, 2, 2, 1};
  int B[N];
  const int M=N-3;
  int ans[M]={1, 3, 2, 1};
  thrust::host_vector<int> h_V(A,A+N);
  thrust::device_vector<int> d_V(A,A+N);
  thrust::host_vector<int> h_result(B,B+M);
  thrust::device_vector<int> d_result(B,B+M);

// CHECK:  oneapi::dpl::unique_copy(oneapi::dpl::execution::seq, h_V.begin(), h_V.end(), h_result.begin());
// CHECK-NEXT:  oneapi::dpl::unique_copy(oneapi::dpl::execution::seq, h_V.begin(), h_V.end(), h_result.begin());
// CHECK-NEXT:  oneapi::dpl::unique_copy(oneapi::dpl::execution::seq, h_V.begin(), h_V.end(), h_result.begin(), oneapi::dpl::equal_to<int>());
// CHECK-NEXT:  oneapi::dpl::unique_copy(oneapi::dpl::execution::seq, h_V.begin(), h_V.end(), h_result.begin(), oneapi::dpl::equal_to<int>());
// CHECK-NEXT:  oneapi::dpl::unique_copy(oneapi::dpl::execution::make_device_policy(q_ct1), d_V.begin(), d_V.end(), d_result.begin());
// CHECK-NEXT:  oneapi::dpl::unique_copy(oneapi::dpl::execution::make_device_policy(q_ct1), d_V.begin(), d_V.end(), d_result.begin());
// CHECK-NEXT:  oneapi::dpl::unique_copy(oneapi::dpl::execution::make_device_policy(q_ct1), d_V.begin(), d_V.end(), d_result.begin(), oneapi::dpl::equal_to<int>());
// CHECK-NEXT:  oneapi::dpl::unique_copy(oneapi::dpl::execution::make_device_policy(q_ct1), d_V.begin(), d_V.end(), d_result.begin(), oneapi::dpl::equal_to<int>());
// CHECK-NEXT:  oneapi::dpl::unique_copy(oneapi::dpl::execution::seq, A, A + N, B);
// CHECK-NEXT:  oneapi::dpl::unique_copy(oneapi::dpl::execution::seq, A, A + N, B);
// CHECK-NEXT:  oneapi::dpl::unique_copy(oneapi::dpl::execution::seq, A, A + N, B, oneapi::dpl::equal_to<int>());
// CHECK-NEXT:  oneapi::dpl::unique_copy(oneapi::dpl::execution::seq, A, A + N, B, oneapi::dpl::equal_to<int>());
  thrust::unique_copy(thrust::host, h_V.begin(), h_V.end(), h_result.begin());
  thrust::unique_copy(h_V.begin(), h_V.end(), h_result.begin());
  thrust::unique_copy(thrust::host, h_V.begin(), h_V.end(), h_result.begin(), thrust::equal_to<int>());
  thrust::unique_copy(h_V.begin(), h_V.end(), h_result.begin(), thrust::equal_to<int>());
  thrust::unique_copy(thrust::device, d_V.begin(), d_V.end(), d_result.begin());
  thrust::unique_copy(d_V.begin(), d_V.end(), d_result.begin());
  thrust::unique_copy(thrust::device, d_V.begin(), d_V.end(), d_result.begin(), thrust::equal_to<int>());
  thrust::unique_copy(d_V.begin(), d_V.end(), d_result.begin(), thrust::equal_to<int>());
  thrust::unique_copy(thrust::host,A, A + N, B);
  thrust::unique_copy(A, A + N, B);
  thrust::unique_copy(thrust::host,A, A + N, B, thrust::equal_to<int>());
  thrust::unique_copy(A, A + N, B, thrust::equal_to<int>());
}

void stable_sort_test() {
  const int N=6;
  int datas[N]={1, 4, 2, 8, 5, 7};
  int ans[N]={1, 2, 4, 5, 7, 8};
  thrust::host_vector<int> h_v(datas,datas+N);
  thrust::device_vector<int> d_v(datas,datas+N);
// CHECK:  oneapi::dpl::stable_sort(oneapi::dpl::execution::seq, h_v.begin(), h_v.end());
// CHECK-NEXT:  oneapi::dpl::stable_sort(oneapi::dpl::execution::seq, h_v.begin(), h_v.end());
// CHECK-NEXT:  oneapi::dpl::stable_sort(oneapi::dpl::execution::seq, h_v.begin(), h_v.end(), std::greater<int>());
// CHECK-NEXT:  oneapi::dpl::stable_sort(oneapi::dpl::execution::seq, h_v.begin(), h_v.end(), std::greater<int>());
// CHECK-NEXT:  oneapi::dpl::stable_sort(oneapi::dpl::execution::make_device_policy(q_ct1), d_v.begin(), d_v.end());
// CHECK-NEXT:  oneapi::dpl::stable_sort(oneapi::dpl::execution::make_device_policy(q_ct1), d_v.begin(), d_v.end());
// CHECK-NEXT:  oneapi::dpl::stable_sort(oneapi::dpl::execution::make_device_policy(q_ct1), d_v.begin(), d_v.end(), std::greater<int>());
// CHECK-NEXT:  oneapi::dpl::stable_sort(oneapi::dpl::execution::make_device_policy(q_ct1), d_v.begin(), d_v.end(), std::greater<int>());
// CHECK-NEXT:  oneapi::dpl::stable_sort(oneapi::dpl::execution::seq, datas, datas+N);
// CHECK-NEXT:  oneapi::dpl::stable_sort(oneapi::dpl::execution::seq, datas, datas+N);
// CHECK-NEXT:  oneapi::dpl::stable_sort(oneapi::dpl::execution::seq, datas, datas+N, std::greater<int>());
// CHECK-NEXT:  oneapi::dpl::stable_sort(oneapi::dpl::execution::seq, datas, datas+N, std::greater<int>());
  thrust::stable_sort(thrust::host, h_v.begin(), h_v.end());
  thrust::stable_sort(h_v.begin(), h_v.end());
  thrust::stable_sort(thrust::host, h_v.begin(), h_v.end(), thrust::greater<int>());
  thrust::stable_sort(h_v.begin(), h_v.end(), thrust::greater<int>());
  thrust::stable_sort(thrust::device, d_v.begin(), d_v.end());
  thrust::stable_sort(d_v.begin(), d_v.end());
  thrust::stable_sort(thrust::device, d_v.begin(), d_v.end(), thrust::greater<int>());
  thrust::stable_sort(d_v.begin(), d_v.end(), thrust::greater<int>());
  thrust::stable_sort(thrust::host, datas,datas+N);
  thrust::stable_sort(datas,datas+N);
  thrust::stable_sort(thrust::host, datas,datas+N, thrust::greater<int>());
  thrust::stable_sort(datas,datas+N, thrust::greater<int>());
}

void set_difference_by_key_test() {
  const int N=7,M=5,P=3;
  int Akey[N]={0, 1, 3, 4, 5, 6, 9};
  int Avalue[N]={0, 0, 0, 0, 0, 0, 0};
  int Bkey[M]={1, 3, 5, 7, 9};
  int Bvalue[N]={1, 1, 1, 1, 1 };

  int Ckey[P];
  int Cvalue[P];
  int anskey[P]={0,4,6};
  int ansvalue[P]={0,0,0};

  thrust::host_vector<int> h_VAkey(Akey,Akey+N);
  thrust::host_vector<int> h_VAvalue(Avalue,Avalue+N);

  thrust::host_vector<int> h_VBkey(Bkey,Bkey+M);
  thrust::host_vector<int> h_VBvalue(Bvalue,Bvalue+M);

  thrust::host_vector<int> h_VCkey(Ckey,Ckey+P);
  thrust::host_vector<int> h_VCvalue(Cvalue,Cvalue+P);
  typedef thrust::pair<thrust::host_vector<int>::iterator, thrust::host_vector<int>::iterator> h_iter_pair;
  thrust::device_vector<int> d_VAkey(Akey,Akey+N);
  thrust::device_vector<int> d_VAvalue(Avalue,Avalue+N);

  thrust::device_vector<int> d_VBkey(Bkey,Bkey+M);
  thrust::device_vector<int> d_VBvalue(Bvalue,Bvalue+M);

  thrust::device_vector<int> d_VCkey(Ckey,Ckey+P);
  thrust::device_vector<int> d_VCvalue(Cvalue,Cvalue+P);
  typedef thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> d_iter_pair;

// CHECK:  dpct::set_difference(oneapi::dpl::execution::seq, h_VAkey.begin(), h_VAkey.end(), h_VBkey.begin(), h_VBkey.end(), h_VAvalue.begin(), h_VBvalue.begin(), h_VCkey.begin(), h_VCvalue.begin());
// CHECK-NEXT:  dpct::set_difference(oneapi::dpl::execution::seq, h_VAkey.begin(), h_VAkey.end(), h_VBkey.begin(), h_VBkey.end(), h_VAvalue.begin(), h_VBvalue.begin(), h_VCkey.begin(), h_VCvalue.begin());
// CHECK-NEXT:  dpct::set_difference(oneapi::dpl::execution::seq, h_VAkey.begin(), h_VAkey.end(), h_VBkey.begin(), h_VBkey.end(), h_VAvalue.begin(), h_VBvalue.begin(), h_VCkey.begin(), h_VCvalue.begin(), std::greater<int>());
// CHECK-NEXT:  dpct::set_difference(oneapi::dpl::execution::seq, h_VAkey.begin(), h_VAkey.end(), h_VBkey.begin(), h_VBkey.end(), h_VAvalue.begin(), h_VBvalue.begin(), h_VCkey.begin(), h_VCvalue.begin(), std::greater<int>());
// CHECK-NEXT:  dpct::set_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_VAkey.begin(), d_VAkey.end(), d_VBkey.begin(), d_VBkey.end(), d_VAvalue.begin(), d_VBvalue.begin(), d_VCkey.begin(), d_VCvalue.begin());
// CHECK-NEXT:  dpct::set_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_VAkey.begin(), d_VAkey.end(), d_VBkey.begin(), d_VBkey.end(), d_VAvalue.begin(), d_VBvalue.begin(), d_VCkey.begin(), d_VCvalue.begin());
// CHECK-NEXT:  dpct::set_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_VAkey.begin(), d_VAkey.end(), d_VBkey.begin(), d_VBkey.end(), d_VAvalue.begin(), d_VBvalue.begin(), d_VCkey.begin(), d_VCvalue.begin(), std::greater<int>());
// CHECK-NEXT:  dpct::set_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_VAkey.begin(), d_VAkey.end(), d_VBkey.begin(), d_VBkey.end(), d_VAvalue.begin(), d_VBvalue.begin(), d_VCkey.begin(), d_VCvalue.begin(), std::greater<int>());
// CHECK-NEXT:  dpct::set_difference(oneapi::dpl::execution::seq, Akey, Akey+N, Bkey, Bkey+M, Avalue, Bvalue, Ckey, Cvalue);
// CHECK-NEXT:  dpct::set_difference(oneapi::dpl::execution::seq, Akey, Akey+N, Bkey, Bkey+M, Avalue, Bvalue, Ckey, Cvalue);
// CHECK-NEXT:  dpct::set_difference(oneapi::dpl::execution::seq, Akey, Akey+N, Bkey, Bkey+M, Avalue, Bvalue, Ckey, Cvalue, std::greater<int>());
// CHECK-NEXT:  dpct::set_difference(oneapi::dpl::execution::seq, Akey, Akey+N, Bkey, Bkey+M, Avalue, Bvalue, Ckey, Cvalue, std::greater<int>());
  thrust::set_difference_by_key(thrust::host, h_VAkey.begin(), h_VAkey.end(), h_VBkey.begin(), h_VBkey.end(), h_VAvalue.begin(), h_VBvalue.begin(), h_VCkey.begin(), h_VCvalue.begin());
  thrust::set_difference_by_key(h_VAkey.begin(), h_VAkey.end(), h_VBkey.begin(), h_VBkey.end(), h_VAvalue.begin(), h_VBvalue.begin(), h_VCkey.begin(), h_VCvalue.begin());
  thrust::set_difference_by_key(thrust::host, h_VAkey.begin(), h_VAkey.end(), h_VBkey.begin(), h_VBkey.end(), h_VAvalue.begin(), h_VBvalue.begin(), h_VCkey.begin(), h_VCvalue.begin(), thrust::greater<int>());
  thrust::set_difference_by_key(h_VAkey.begin(), h_VAkey.end(), h_VBkey.begin(), h_VBkey.end(), h_VAvalue.begin(), h_VBvalue.begin(), h_VCkey.begin(), h_VCvalue.begin(), thrust::greater<int>());
  thrust::set_difference_by_key(thrust::device, d_VAkey.begin(), d_VAkey.end(), d_VBkey.begin(), d_VBkey.end(), d_VAvalue.begin(), d_VBvalue.begin(), d_VCkey.begin(), d_VCvalue.begin());
  thrust::set_difference_by_key(d_VAkey.begin(), d_VAkey.end(), d_VBkey.begin(), d_VBkey.end(), d_VAvalue.begin(), d_VBvalue.begin(), d_VCkey.begin(), d_VCvalue.begin());
  thrust::set_difference_by_key(thrust::device, d_VAkey.begin(), d_VAkey.end(), d_VBkey.begin(), d_VBkey.end(), d_VAvalue.begin(), d_VBvalue.begin(), d_VCkey.begin(), d_VCvalue.begin(),thrust::greater<int>());
  thrust::set_difference_by_key(d_VAkey.begin(), d_VAkey.end(), d_VBkey.begin(), d_VBkey.end(), d_VAvalue.begin(), d_VBvalue.begin(), d_VCkey.begin(), d_VCvalue.begin(), thrust::greater<int>());
  thrust::set_difference_by_key(thrust::host,Akey,Akey+N,Bkey,Bkey+M,Avalue,Bvalue,Ckey,Cvalue);
  thrust::set_difference_by_key(Akey,Akey+N,Bkey,Bkey+M,Avalue,Bvalue,Ckey,Cvalue);
  thrust::set_difference_by_key(thrust::host,Akey,Akey+N,Bkey,Bkey+M,Avalue,Bvalue,Ckey,Cvalue, thrust::greater<int>());
  thrust::set_difference_by_key(Akey,Akey+N,Bkey,Bkey+M,Avalue,Bvalue,Ckey,Cvalue, thrust::greater<int>());
}

void set_difference_test() {
  const int N=7,M=5,P=3;
  int A[N]={0, 1, 3, 4, 5, 6, 9};
  int B[M]={1, 3, 5, 7, 9};
  int C[P];
  int ans[P]={0,4,6};
  thrust::host_vector<int> h_VA(A,A+N);
  thrust::host_vector<int> h_VB(B,B+M);
  thrust::host_vector<int> h_VC(C,C+P);
  thrust::device_vector<int> d_VA(A,A+N);
  thrust::device_vector<int> d_VB(B,B+M);
  thrust::device_vector<int> d_VC(C,C+P);

// CHECK:  oneapi::dpl::set_difference(oneapi::dpl::execution::seq, h_VA.begin(), h_VA.end(), h_VB.begin(), h_VB.end(), h_VC.begin());
// CHECK-NEXT:  oneapi::dpl::set_difference(oneapi::dpl::execution::seq, h_VA.begin(), h_VA.end(), h_VB.begin(), h_VB.end(), h_VC.begin());
// CHECK-NEXT:  oneapi::dpl::set_difference(oneapi::dpl::execution::seq, h_VA.begin(), h_VA.end(), h_VB.begin(), h_VB.end(), h_VC.begin(), std::greater<int>());
// CHECK-NEXT:  oneapi::dpl::set_difference(oneapi::dpl::execution::seq, h_VA.begin(), h_VA.end(), h_VB.begin(), h_VB.end(), h_VC.begin(), std::greater<int>());
// CHECK-NEXT:  oneapi::dpl::set_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_VA.begin(), d_VA.end(), d_VB.begin(), d_VB.end(), d_VC.begin());
// CHECK-NEXT:  oneapi::dpl::set_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_VA.begin(), d_VA.end(), d_VB.begin(), d_VB.end(), d_VC.begin());
// CHECK-NEXT:  oneapi::dpl::set_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_VA.begin(), d_VA.end(), d_VB.begin(), d_VB.end(), d_VC.begin(), std::greater<int>());
// CHECK-NEXT:  oneapi::dpl::set_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_VA.begin(), d_VA.end(), d_VB.begin(), d_VB.end(), d_VC.begin(), std::greater<int>());
// CHECK-NEXT:  oneapi::dpl::set_difference(oneapi::dpl::execution::seq, A, A+N, B, B+M, C);
// CHECK-NEXT:  oneapi::dpl::set_difference(oneapi::dpl::execution::seq, A, A+N, B, B+M, C);
// CHECK-NEXT:  oneapi::dpl::set_difference(oneapi::dpl::execution::seq, A, A+N, B, B+M, C, std::greater<int>());
// CHECK-NEXT:  oneapi::dpl::set_difference(oneapi::dpl::execution::seq, A, A+N, B, B+M, C, std::greater<int>());
  thrust::set_difference(thrust::host, h_VA.begin(), h_VA.end(), h_VB.begin(), h_VB.end(), h_VC.begin());
  thrust::set_difference(h_VA.begin(), h_VA.end(), h_VB.begin(), h_VB.end(), h_VC.begin());
  thrust::set_difference(thrust::host, h_VA.begin(), h_VA.end(), h_VB.begin(), h_VB.end(), h_VC.begin(), thrust::greater<int>());
  thrust::set_difference(h_VA.begin(), h_VA.end(), h_VB.begin(), h_VB.end(), h_VC.begin(), thrust::greater<int>());
  thrust::set_difference(thrust::device, d_VA.begin(), d_VA.end(), d_VB.begin(), d_VB.end(), d_VC.begin());
  thrust::set_difference(d_VA.begin(), d_VA.end(), d_VB.begin(), d_VB.end(), d_VC.begin());
  thrust::set_difference(thrust::device, d_VA.begin(), d_VA.end(), d_VB.begin(), d_VB.end(), d_VC.begin(), thrust::greater<int>());
  thrust::set_difference( d_VA.begin(), d_VA.end(), d_VB.begin(), d_VB.end(), d_VC.begin(), thrust::greater<int>());
  thrust::set_difference(thrust::host, A,A+N,B,B+M,C);
  thrust::set_difference( A,A+N,B,B+M,C);
  thrust::set_difference(thrust::host, A,A+N,B,B+M,C, thrust::greater<int>());
  thrust::set_difference( A,A+N,B,B+M,C, thrust::greater<int>());
}


struct add_functor
{
  __host__ __device__
  void operator()(int & x)
  {
    x++;
  }
};
void for_each_n_test() {
  const int N=3;
  int A[N]={0,1,2};
  int ans[N]={1,2,3};
  thrust::host_vector<int> h_V(A,A+N);
  thrust::device_vector<int> d_V(A,A+N);

// CHECK:  oneapi::dpl::for_each_n(oneapi::dpl::execution::seq, h_V.begin(), h_V.size(), add_functor());
// CHECK-NEXT:  oneapi::dpl::for_each_n(oneapi::dpl::execution::seq, h_V.begin(), h_V.size(), add_functor());
// CHECK-NEXT:  oneapi::dpl::for_each_n(oneapi::dpl::execution::make_device_policy(q_ct1), d_V.begin(), d_V.size(), add_functor());
// CHECK-NEXT:  oneapi::dpl::for_each_n(oneapi::dpl::execution::seq, d_V.begin(), d_V.size(), add_functor());
// CHECK-NEXT:  oneapi::dpl::for_each_n(oneapi::dpl::execution::seq, A, N, add_functor());
// CHECK-NEXT:  oneapi::dpl::for_each_n(oneapi::dpl::execution::seq, A, N, add_functor());
  thrust::for_each_n(thrust::host, h_V.begin(), h_V.size(), add_functor());
  thrust::for_each_n(h_V.begin(), h_V.size(), add_functor());
  thrust::for_each_n(thrust::device, d_V.begin(), d_V.size(), add_functor());
  thrust::for_each_n(d_V.begin(), d_V.size(), add_functor());
  thrust::for_each_n(thrust::host, A, N, add_functor());
  thrust::for_each_n(A, N, add_functor());
}


void tabulate_test() {
  const int N=10;
  int A[N];
  int ans[N]={0, -1, -2, -3, -4, -5, -6, -7, -8, -9};
  thrust::host_vector<int> h_V(A,A+N);
  thrust::device_vector<int> d_V(A,A+N);

// CHECK:  dpct::for_each_index(oneapi::dpl::execution::seq, h_V.begin(), h_V.end(), std::negate<int>());
// CHECK-NEXT:  dpct::for_each_index(oneapi::dpl::execution::seq, h_V.begin(), h_V.end(), std::negate<int>());
// CHECK-NEXT:  dpct::for_each_index(oneapi::dpl::execution::make_device_policy(q_ct1), d_V.begin(), d_V.end(), std::negate<int>());
// CHECK-NEXT:  dpct::for_each_index(oneapi::dpl::execution::make_device_policy(q_ct1), d_V.begin(), d_V.end(), std::negate<int>());
// CHECK-NEXT:  dpct::for_each_index(oneapi::dpl::execution::seq, A, A+N, std::negate<int>());
// CHECK-NEXT:  dpct::for_each_index(oneapi::dpl::execution::seq, A, A+N, std::negate<int>());
  thrust::tabulate(thrust::host, h_V.begin(), h_V.end(), thrust::negate<int>());
  thrust::tabulate( h_V.begin(), h_V.end(), thrust::negate<int>());
  thrust::tabulate(thrust::device, d_V.begin(), d_V.end(), thrust::negate<int>());
  thrust::tabulate(d_V.begin(), d_V.end(), thrust::negate<int>());
  thrust::tabulate(thrust::host, A,A+N, thrust::negate<int>());
  thrust::tabulate(A,A+N, thrust::negate<int>());
}

void remove_copy_test() {
  const int N = 6;
  int A[N] = {-2, 0, -1, 0, 1, 2};
  int B[N - 2];
  int ans[N - 2] = {-2, -1, 1, 2};
  int result[N - 2];
  int V[N] = {-2, 0, -1, 0, 1, 2};

  thrust::host_vector<int> h_V(A, A + N);
  thrust::host_vector<int> h_result(B, B + N - 2);
  thrust::device_vector<int> d_V(A, A + N);
  thrust::device_vector<int> d_result(B, B + N - 2);

// CHECK:  oneapi::dpl::remove_copy(oneapi::dpl::execution::seq, h_V.begin(), h_V.end(), h_result.begin(), 0);
// CHECK-NEXT:  oneapi::dpl::remove_copy(oneapi::dpl::execution::seq, h_V.begin(), h_V.end(), h_result.begin(), 0);
// CHECK-NEXT:  oneapi::dpl::remove_copy(oneapi::dpl::execution::make_device_policy(q_ct1), d_V.begin(), d_V.end(), d_result.begin(), 0);
// CHECK-NEXT:  oneapi::dpl::remove_copy(oneapi::dpl::execution::make_device_policy(q_ct1), d_V.begin(), d_V.end(), d_result.begin(), 0);
// CHECK-NEXT:  oneapi::dpl::remove_copy(oneapi::dpl::execution::seq, V, V + N, result, 0);
// CHECK-NEXT:  oneapi::dpl::remove_copy(oneapi::dpl::execution::seq, V, V + N, result, 0);
  thrust::remove_copy(thrust::host, h_V.begin(), h_V.end(), h_result.begin(), 0);
  thrust::remove_copy(h_V.begin(), h_V.end(), h_result.begin(), 0);
  thrust::remove_copy(thrust::device, d_V.begin(), d_V.end(), d_result.begin(), 0);
  thrust::remove_copy(d_V.begin(), d_V.end(), d_result.begin(), 0);
  thrust::remove_copy(thrust::host, V, V + N, result, 0);
  thrust::remove_copy(V, V + N, result, 0);
}

void transform_exclusive_scan_test() {
  const int N=6;
  int A[N]={1, 0, 2, 2, 1, 3};
  int ans[N]={4, 3, 3, 1, -1, -2};
  thrust::host_vector<int> h_V(A,A+N);
  thrust::device_vector<int> d_V(A,A+N);
  thrust::negate<int> unary_op;
  thrust::plus<int> binary_op;

// CHECK:  oneapi::dpl::transform_exclusive_scan(oneapi::dpl::execution::seq, h_V.begin(), h_V.end(), h_V.begin(), 4, binary_op, unary_op);
// CHECK-NEXT:  oneapi::dpl::transform_exclusive_scan(oneapi::dpl::execution::seq, h_V.begin(), h_V.end(), h_V.begin(), 4, binary_op, unary_op);
// CHECK-NEXT:  oneapi::dpl::transform_exclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), d_V.begin(), d_V.end(), d_V.begin(), 4, binary_op, unary_op);
// CHECK-NEXT:  oneapi::dpl::transform_exclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), d_V.begin(), d_V.end(), d_V.begin(), 4, binary_op, unary_op);
// CHECK-NEXT:  oneapi::dpl::transform_exclusive_scan(oneapi::dpl::execution::seq, A, A+N, A, 4, binary_op, unary_op);
// CHECK-NEXT: oneapi::dpl::transform_exclusive_scan(oneapi::dpl::execution::seq, A, A+N, A, 4, binary_op, unary_op);
  thrust::transform_exclusive_scan(thrust::host, h_V.begin(), h_V.end(), h_V.begin(), unary_op, 4, binary_op);
  thrust::transform_exclusive_scan(h_V.begin(), h_V.end(), h_V.begin(), unary_op, 4, binary_op);
  thrust::transform_exclusive_scan(thrust::device, d_V.begin(), d_V.end(), d_V.begin(), unary_op, 4, binary_op);
  thrust::transform_exclusive_scan(d_V.begin(), d_V.end(), d_V.begin(), unary_op, 4, binary_op);
  thrust::transform_exclusive_scan(thrust::host, A, A+N, A, unary_op, 4, binary_op);
  thrust::transform_exclusive_scan(A, A+N, A, unary_op, 4, binary_op);
}


void set_intersection_by_key_test() {
  const int N = 6, M = 7, P = 3;
  int Akey[N] = {1, 3, 5, 7, 9, 11};
  int Avalue[N] = {0, 0, 0, 0, 0, 0};
  int Bkey[M] = {1, 1, 2, 3, 5, 8, 13};

  int Ckey[P];
  int Cvalue[P];
  int anskey[P] = {1, 3, 5};
  int ansvalue[P] = {0, 0, 0};

  thrust::host_vector<int> h_VAkey(Akey, Akey + N);
  thrust::host_vector<int> h_VAvalue(Avalue, Avalue + N);

  thrust::host_vector<int> h_VBkey(Bkey, Bkey + M);

  thrust::host_vector<int> h_VCkey(Ckey, Ckey + P);
  thrust::host_vector<int> h_VCvalue(Cvalue, Cvalue + P);
  typedef thrust::pair<thrust::host_vector<int>::iterator,
                       thrust::host_vector<int>::iterator> iter_pair;
  thrust::device_vector<int> d_VAkey(Akey, Akey + N);
  thrust::device_vector<int> d_VAvalue(Avalue, Avalue + N);
  thrust::device_vector<int> d_VBkey(Bkey, Bkey + M);
  thrust::device_vector<int> d_VCkey(Ckey, Ckey + P);
  thrust::device_vector<int> d_VCvalue(Cvalue, Cvalue + P);


// CHECK:  dpct::set_intersection(oneapi::dpl::execution::seq, h_VAkey.begin(), h_VAkey.end(), h_VBkey.begin(), h_VBkey.end(), h_VAvalue.begin(), h_VCkey.begin(), h_VCvalue.begin());
// CHECK-NEXT:  dpct::set_intersection(oneapi::dpl::execution::seq, h_VAkey.begin(), h_VAkey.end(), h_VBkey.begin(), h_VBkey.end(), h_VAvalue.begin(), h_VCkey.begin(), h_VCvalue.begin());
// CHECK-NEXT:  dpct::set_intersection(oneapi::dpl::execution::seq, h_VAkey.begin(), h_VAkey.end(), h_VBkey.begin(), h_VBkey.end(), h_VAvalue.begin(), h_VCkey.begin(), h_VCvalue.begin(), std::greater<int>());
// CHECK-NEXT:  dpct::set_intersection(oneapi::dpl::execution::seq, h_VAkey.begin(), h_VAkey.end(), h_VBkey.begin(), h_VBkey.end(), h_VAvalue.begin(), h_VCkey.begin(), h_VCvalue.begin(), std::greater<int>());
// CHECK-NEXT:  dpct::set_intersection(oneapi::dpl::execution::make_device_policy(q_ct1), d_VAkey.begin(), d_VAkey.end(), d_VBkey.begin(), d_VBkey.end(), d_VAvalue.begin(), d_VCkey.begin(), d_VCvalue.begin());
// CHECK-NEXT:  dpct::set_intersection(oneapi::dpl::execution::make_device_policy(q_ct1), d_VAkey.begin(), d_VAkey.end(), d_VBkey.begin(), d_VBkey.end(), d_VAvalue.begin(), d_VCkey.begin(), d_VCvalue.begin());
// CHECK-NEXT:  dpct::set_intersection(oneapi::dpl::execution::make_device_policy(q_ct1), d_VAkey.begin(), d_VAkey.end(), d_VBkey.begin(), d_VBkey.end(), d_VAvalue.begin(), d_VCkey.begin(), d_VCvalue.begin(), std::greater<int>());
// CHECK-NEXT:  dpct::set_intersection(oneapi::dpl::execution::make_device_policy(q_ct1), d_VAkey.begin(), d_VAkey.end(), d_VBkey.begin(), d_VBkey.end(), d_VAvalue.begin(), d_VCkey.begin(), d_VCvalue.begin(), std::greater<int>());
// CHECK-NEXT:  dpct::set_intersection(oneapi::dpl::execution::seq, Akey, Akey + N, Bkey, Bkey + M, Avalue, Ckey, Cvalue);
// CHECK-NEXT:  dpct::set_intersection(oneapi::dpl::execution::seq, Akey, Akey + N, Bkey, Bkey + M, Avalue, Ckey, Cvalue);
// CHECK-NEXT:  dpct::set_intersection(oneapi::dpl::execution::seq, Akey, Akey + N, Bkey, Bkey + M, Avalue, Ckey, Cvalue, std::greater<int>());
// CHECK-NEXT:  dpct::set_intersection(oneapi::dpl::execution::seq, Akey, Akey + N, Bkey, Bkey + M, Avalue, Ckey, Cvalue, std::greater<int>());
  thrust::set_intersection_by_key(thrust::host, h_VAkey.begin(), h_VAkey.end(), h_VBkey.begin(), h_VBkey.end(), h_VAvalue.begin(), h_VCkey.begin(), h_VCvalue.begin());
  thrust::set_intersection_by_key(h_VAkey.begin(), h_VAkey.end(), h_VBkey.begin(), h_VBkey.end(), h_VAvalue.begin(), h_VCkey.begin(), h_VCvalue.begin());
  thrust::set_intersection_by_key(thrust::host, h_VAkey.begin(), h_VAkey.end(), h_VBkey.begin(), h_VBkey.end(), h_VAvalue.begin(), h_VCkey.begin(), h_VCvalue.begin(), thrust::greater<int>());
  thrust::set_intersection_by_key(h_VAkey.begin(), h_VAkey.end(), h_VBkey.begin(), h_VBkey.end(), h_VAvalue.begin(), h_VCkey.begin(), h_VCvalue.begin(), thrust::greater<int>());
  thrust::set_intersection_by_key(thrust::device, d_VAkey.begin(), d_VAkey.end(), d_VBkey.begin(), d_VBkey.end(), d_VAvalue.begin(), d_VCkey.begin(), d_VCvalue.begin());
  thrust::set_intersection_by_key(d_VAkey.begin(), d_VAkey.end(), d_VBkey.begin(), d_VBkey.end(), d_VAvalue.begin(), d_VCkey.begin(), d_VCvalue.begin());
  thrust::set_intersection_by_key(thrust::device, d_VAkey.begin(), d_VAkey.end(), d_VBkey.begin(), d_VBkey.end(), d_VAvalue.begin(), d_VCkey.begin(), d_VCvalue.begin(), thrust::greater<int>());
  thrust::set_intersection_by_key(d_VAkey.begin(), d_VAkey.end(), d_VBkey.begin(), d_VBkey.end(), d_VAvalue.begin(), d_VCkey.begin(), d_VCvalue.begin(), thrust::greater<int>());
  thrust::set_intersection_by_key(thrust::host, Akey, Akey + N, Bkey, Bkey + M, Avalue, Ckey, Cvalue);
  thrust::set_intersection_by_key(Akey, Akey + N, Bkey, Bkey + M, Avalue, Ckey, Cvalue);
  thrust::set_intersection_by_key(thrust::host, Akey, Akey + N, Bkey, Bkey + M, Avalue, Ckey, Cvalue, thrust::greater<int>());
  thrust::set_intersection_by_key(Akey, Akey + N, Bkey, Bkey + M, Avalue, Ckey, Cvalue, thrust::greater<int>());
}

void raw_reference_cast_test() {
  thrust::host_vector<int> h_vec(1);
  thrust::device_vector<int> d_vec = h_vec;
  const thrust::device_reference<int> ref_const = d_vec[0];
// CHECK:  int &ref1 = dpct::get_raw_reference(d_vec[0]);
// CHECK-NEXT:  int &ref2 = dpct::get_raw_reference(ref_const);
  int &ref1 = thrust::raw_reference_cast(d_vec[0]);
  int &ref2 = thrust::raw_reference_cast(ref_const);
}

void partition_copy_test() {
  int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int S[] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  int result[10];
  const int N = sizeof(data) / sizeof(int);
  int *evens = result;
  int *odds = result + N / 2;

  thrust::host_vector<int> host_a(data, data + N);
  thrust::host_vector<int> host_evens(N / 2);
  thrust::host_vector<int> host_odds(N / 2);

  thrust::device_vector<int> device_a(data, data + N);
  thrust::device_vector<int> device_evens(N / 2);
  thrust::device_vector<int> device_odds(N / 2);
  thrust::host_vector<int> host_S(S, S + N);
  thrust::device_vector<int> device_S(S, S + N);


// CHECK:    oneapi::dpl::partition_copy(oneapi::dpl::execution::seq, data, data + N, evens, odds, is_even());
// CHECK-NEXT:  oneapi::dpl::partition_copy(oneapi::dpl::execution::seq, host_a.begin(), host_a.begin() + N, host_evens.begin(), host_odds.begin(), is_even());
// CHECK-NEXT:  oneapi::dpl::partition_copy(oneapi::dpl::execution::make_device_policy(q_ct1), device_a.begin(), device_a.begin() + N, device_evens.begin(), device_odds.begin(), is_even());
// CHECK-NEXT:  oneapi::dpl::partition_copy(oneapi::dpl::execution::seq, data, data + N, evens, odds, is_even());
// CHECK-NEXT:  oneapi::dpl::partition_copy(oneapi::dpl::execution::seq, host_a.begin(), host_a.begin() + N, host_evens.begin(), host_odds.begin(), is_even());
// CHECK-NEXT:  oneapi::dpl::partition_copy(oneapi::dpl::execution::make_device_policy(q_ct1), device_a.begin(), device_a.begin() + N, device_evens.begin(), device_odds.begin(), is_even());
// CHECK-NEXT:  dpct::partition_copy(oneapi::dpl::execution::seq, data, data + N, S, evens, odds, is_even());
// CHECK-NEXT:  dpct::partition_copy(oneapi::dpl::execution::seq, host_a.begin(), host_a.begin() + N, host_S.begin(), host_evens.begin(), host_odds.begin(), is_even());
// CHECK-NEXT:  dpct::partition_copy(oneapi::dpl::execution::make_device_policy(q_ct1), device_a.begin(), device_a.begin() + N, device_S.begin(), device_evens.begin(), device_odds.begin(), is_even());
// CHECK-NEXT:  dpct::partition_copy(oneapi::dpl::execution::seq, data, data + N, S, evens, odds, is_even());
// CHECK-NEXT:  dpct::partition_copy(oneapi::dpl::execution::seq, host_a.begin(), host_a.begin() + N, host_S.begin(), host_evens.begin(), host_odds.begin(), is_even());
// CHECK-NEXT:  dpct::partition_copy(oneapi::dpl::execution::make_device_policy(q_ct1), device_a.begin(), device_a.begin() + N, device_S.begin(), device_evens.begin(), device_odds.begin(), is_even());
  thrust::partition_copy(thrust::host, data, data + N, evens, odds, is_even());
  thrust::partition_copy(thrust::host, host_a.begin(), host_a.begin() + N, host_evens.begin(), host_odds.begin(), is_even());
  thrust::partition_copy(thrust::device, device_a.begin(), device_a.begin() + N, device_evens.begin(), device_odds.begin(), is_even());
  thrust::partition_copy(data, data + N, evens, odds, is_even());
  thrust::partition_copy(host_a.begin(), host_a.begin() + N, host_evens.begin(), host_odds.begin(), is_even());
  thrust::partition_copy(device_a.begin(), device_a.begin() + N, device_evens.begin(), device_odds.begin(), is_even());
  thrust::partition_copy(thrust::host, data, data + N, S, evens, odds, is_even());
  thrust::partition_copy(thrust::host, host_a.begin(), host_a.begin() + N, host_S.begin(), host_evens.begin(), host_odds.begin(), is_even());
  thrust::partition_copy(thrust::device, device_a.begin(), device_a.begin() + N, device_S.begin(), device_evens.begin(), device_odds.begin(), is_even());
  thrust::partition_copy(data, data + N, S, evens, odds, is_even());
  thrust::partition_copy(host_a.begin(), host_a.begin() + N, host_S.begin(), host_evens.begin(), host_odds.begin(), is_even());
  thrust::partition_copy(device_a.begin(), device_a.begin() + N, device_S.begin(), device_evens.begin(), device_odds.begin(), is_even());
}

void stable_partition_copy_test() {
  int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int S[] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  int result[10];
  const int N = sizeof(data) / sizeof(int);
  int *evens = result;
  int *odds = result + N / 2;

  thrust::host_vector<int> host_a(data, data + N);
  thrust::host_vector<int> host_evens(N / 2);
  thrust::host_vector<int> host_odds(N / 2);

  thrust::device_vector<int> device_a(data, data + N);
  thrust::device_vector<int> device_evens(N / 2);
  thrust::device_vector<int> device_odds(N / 2);
  thrust::host_vector<int> host_S(S, S + N);
  thrust::device_vector<int> device_S(S, S + N);


// CHECK:    dpct::stable_partition_copy(oneapi::dpl::execution::seq, data, data + N, evens, odds, is_even());
// CHECK-NEXT:  dpct::stable_partition_copy(oneapi::dpl::execution::seq, host_a.begin(), host_a.begin() + N, host_evens.begin(), host_odds.begin(), is_even());
// CHECK-NEXT:  dpct::stable_partition_copy(oneapi::dpl::execution::make_device_policy(q_ct1), device_a.begin(), device_a.begin() + N, device_evens.begin(), device_odds.begin(), is_even());
// CHECK-NEXT:  dpct::stable_partition_copy(oneapi::dpl::execution::seq, data, data + N, evens, odds, is_even());
// CHECK-NEXT:  dpct::stable_partition_copy(oneapi::dpl::execution::seq, host_a.begin(), host_a.begin() + N, host_evens.begin(), host_odds.begin(), is_even());
// CHECK-NEXT:  dpct::stable_partition_copy(oneapi::dpl::execution::make_device_policy(q_ct1), device_a.begin(), device_a.begin() + N, device_evens.begin(), device_odds.begin(), is_even());
// CHECK-NEXT:  dpct::stable_partition_copy(oneapi::dpl::execution::seq, data, data + N, S, evens, odds, is_even());
// CHECK-NEXT:  dpct::stable_partition_copy(oneapi::dpl::execution::seq, host_a.begin(), host_a.begin() + N, host_S.begin(), host_evens.begin(), host_odds.begin(), is_even());
// CHECK-NEXT:  dpct::stable_partition_copy(oneapi::dpl::execution::make_device_policy(q_ct1), device_a.begin(), device_a.begin() + N, device_S.begin(), device_evens.begin(), device_odds.begin(), is_even());
// CHECK-NEXT:  dpct::stable_partition_copy(oneapi::dpl::execution::seq, data, data + N, S, evens, odds, is_even());
// CHECK-NEXT:  dpct::stable_partition_copy(oneapi::dpl::execution::seq, host_a.begin(), host_a.begin() + N, host_S.begin(), host_evens.begin(), host_odds.begin(), is_even());
// CHECK-NEXT:  dpct::stable_partition_copy(oneapi::dpl::execution::make_device_policy(q_ct1), device_a.begin(), device_a.begin() + N, device_S.begin(), device_evens.begin(), device_odds.begin(), is_even());
  thrust::stable_partition_copy(thrust::host, data, data + N, evens, odds, is_even());
  thrust::stable_partition_copy(thrust::host, host_a.begin(), host_a.begin() + N, host_evens.begin(), host_odds.begin(), is_even());
  thrust::stable_partition_copy(thrust::device, device_a.begin(), device_a.begin() + N, device_evens.begin(), device_odds.begin(), is_even());
  thrust::stable_partition_copy(data, data + N, evens, odds, is_even());
  thrust::stable_partition_copy(host_a.begin(), host_a.begin() + N, host_evens.begin(), host_odds.begin(), is_even());
  thrust::stable_partition_copy(device_a.begin(), device_a.begin() + N, device_evens.begin(), device_odds.begin(), is_even());
  thrust::stable_partition_copy(thrust::host, data, data + N, S, evens, odds, is_even());
  thrust::stable_partition_copy(thrust::host, host_a.begin(), host_a.begin() + N, host_S.begin(), host_evens.begin(), host_odds.begin(), is_even());
  thrust::stable_partition_copy(thrust::device, device_a.begin(), device_a.begin() + N, device_S.begin(), device_evens.begin(), device_odds.begin(), is_even());
  thrust::stable_partition_copy(data, data + N, S, evens, odds, is_even());
  thrust::stable_partition_copy(host_a.begin(), host_a.begin() + N, host_S.begin(), host_evens.begin(), host_odds.begin(), is_even());
  thrust::stable_partition_copy(device_a.begin(), device_a.begin() + N, device_S.begin(), device_evens.begin(), device_odds.begin(), is_even());
}
void stable_partition_test() {

  int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int S[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const int N = sizeof(data) / sizeof(int);
  thrust::host_vector<int> host_data(data, data + N);
  thrust::device_vector<int> device_data(data, data + N);
  thrust::host_vector<int> host_S(S, S + N);
  thrust::device_vector<int> device_s(S, S + N);


// CHECK:  oneapi::dpl::stable_partition(oneapi::dpl::execution::seq, data, data + N, is_even());
// CHECK-NEXT:  oneapi::dpl::stable_partition(oneapi::dpl::execution::seq, host_data.begin(), host_data.begin() + N, is_even());
// CHECK-NEXT:  oneapi::dpl::stable_partition(oneapi::dpl::execution::make_device_policy(q_ct1), device_data.begin(), device_data.begin() + N, is_even());
// CHECK-NEXT:  oneapi::dpl::stable_partition(oneapi::dpl::execution::seq, data, data + N, is_even());
// CHECK-NEXT:  oneapi::dpl::stable_partition(oneapi::dpl::execution::seq, host_data.begin(), host_data.begin() + N, is_even());
// CHECK-NEXT:  oneapi::dpl::stable_partition(oneapi::dpl::execution::make_device_policy(q_ct1), device_data.begin(), device_data.begin() + N, is_even());
// CHECK-NEXT:  dpct::stable_partition(oneapi::dpl::execution::seq, data, data + N, S, is_even());
// CHECK-NEXT:  dpct::stable_partition(oneapi::dpl::execution::seq, host_data.begin(), host_data.begin() + N, host_S.begin(), is_even());
// CHECK-NEXT:  dpct::stable_partition(oneapi::dpl::execution::make_device_policy(q_ct1), device_data.begin(), device_data.begin() + N, device_s.begin(), is_even());
// CHECK-NEXT:  dpct::stable_partition(oneapi::dpl::execution::seq, data, data + N, S, is_even());
// CHECK-NEXT:  dpct::stable_partition(oneapi::dpl::execution::seq, host_data.begin(), host_data.begin() + N, host_S.begin(), is_even());
// CHECK-NEXT:  dpct::stable_partition(oneapi::dpl::execution::make_device_policy(q_ct1), device_data.begin(), device_data.begin() + N, device_s.begin(), is_even());
  thrust::stable_partition(thrust::host, data, data + N, is_even());
  thrust::stable_partition(thrust::host, host_data.begin(), host_data.begin() + N, is_even());
  thrust::stable_partition(thrust::device, device_data.begin(), device_data.begin() + N, is_even());
  thrust::stable_partition(data, data + N, is_even());
  thrust::stable_partition(host_data.begin(), host_data.begin() + N, is_even());
  thrust::stable_partition(device_data.begin(), device_data.begin() + N, is_even());
  thrust::stable_partition(thrust::host, data, data + N, S, is_even());
  thrust::stable_partition(thrust::host, host_data.begin(), host_data.begin() + N, host_S.begin(), is_even());
  thrust::stable_partition(thrust::device, device_data.begin(), device_data.begin() + N, device_s.begin(), is_even());
  thrust::stable_partition(data, data + N, S, is_even());
  thrust::stable_partition(host_data.begin(), host_data.begin() + N, host_S.begin(), is_even());
  thrust::stable_partition(device_data.begin(), device_data.begin() + N, device_s.begin(), is_even());
}

void remvoe_test() {
  const int N = 6;
  int data[N] = {3, 1, 4, 1, 5, 9};
  thrust::device_vector<int> device_data(data, data + N);
  thrust::host_vector<int> host_data(data, data + N);

// CHECK:  oneapi::dpl::remove(oneapi::dpl::execution::seq, data, data + N, 1);
// CHECK-NEXT:  oneapi::dpl::remove(oneapi::dpl::execution::seq, host_data.begin(), host_data.begin() + N, 1);
// CHECK-NEXT:  oneapi::dpl::remove(oneapi::dpl::execution::make_device_policy(q_ct1), device_data.begin(), device_data.begin() + N, 1);
// CHECK-NEXT:  oneapi::dpl::remove(oneapi::dpl::execution::seq, data, data + N, 1);
// CHECK-NEXT:  oneapi::dpl::remove(oneapi::dpl::execution::seq, host_data.begin(), host_data.begin() + N, 1);
// CHECK-NEXT:  oneapi::dpl::remove(oneapi::dpl::execution::make_device_policy(q_ct1), device_data.begin(), device_data.begin() + N, 1);
  thrust::remove(thrust::host, data, data + N, 1);
  thrust::remove(thrust::host, host_data.begin(), host_data.begin() + N, 1);
  thrust::remove(thrust::device, device_data.begin(), device_data.begin() + N, 1);
  thrust::remove(data, data + N, 1);
  thrust::remove(host_data.begin(), host_data.begin() + N, 1);
  thrust::remove(device_data.begin(), device_data.begin() + N, 1);
}

struct greater_than_four {
  __host__ __device__ bool operator()(int x) const { return x > 4; }
};

void find_if_test() {
  const int N = 4;
  int data[4] = {0,5, 3, 7};
  thrust::device_vector<int> device_data(data, data + N);
  thrust::host_vector<int> host_data(data, data + N);

  // CHECK:oneapi::dpl::find_if(oneapi::dpl::execution::seq, data, data+3, greater_than_four());
  // CHECK-NEXT:oneapi::dpl::find_if(oneapi::dpl::execution::make_device_policy(q_ct1), device_data.begin(), device_data.end(), greater_than_four());
  // CHECK-NEXT:oneapi::dpl::find_if(oneapi::dpl::execution::seq, host_data.begin(), host_data.end(), greater_than_four());
  // CHECK-NEXT:oneapi::dpl::find_if(oneapi::dpl::execution::seq, data, data+3, greater_than_four());
  // CHECK-NEXT:oneapi::dpl::find_if(oneapi::dpl::execution::make_device_policy(q_ct1), device_data.begin(), device_data.end(), greater_than_four());
  // CHECK-NEXT:oneapi::dpl::find_if(oneapi::dpl::execution::seq, host_data.begin(), host_data.end(), greater_than_four());
  thrust::find_if(data, data+3, greater_than_four());
  thrust::find_if(device_data.begin(), device_data.end(),  greater_than_four());
  thrust::find_if(host_data.begin(), host_data.end(),  greater_than_four());
  thrust::find_if(thrust::host, data, data+3, greater_than_four());
  thrust::find_if(thrust::device, device_data.begin(), device_data.end(),  greater_than_four());
  thrust::find_if(thrust::host, host_data.begin(), host_data.end(),  greater_than_four());
}

void find_if_not_test() {
  const int N = 4;
  int data[4] = {0,5, 3, 7};
  thrust::device_vector<int> device_data(data, data + N);
  thrust::host_vector<int> host_data(data, data + N);

  // CHECK:oneapi::dpl::find_if_not(oneapi::dpl::execution::seq, data, data+3, greater_than_four());
  // CHECK-NEXT:oneapi::dpl::find_if_not(oneapi::dpl::execution::make_device_policy(q_ct1), device_data.begin(), device_data.end(), greater_than_four());
  // CHECK-NEXT:oneapi::dpl::find_if_not(oneapi::dpl::execution::seq, host_data.begin(), host_data.end(), greater_than_four());
  // CHECK-NEXT:oneapi::dpl::find_if_not(oneapi::dpl::execution::seq, data, data+3, greater_than_four());
  // CHECK-NEXT:oneapi::dpl::find_if_not(oneapi::dpl::execution::make_device_policy(q_ct1), device_data.begin(), device_data.end(), greater_than_four());
  // CHECK-NEXT:oneapi::dpl::find_if_not(oneapi::dpl::execution::seq, host_data.begin(), host_data.end(), greater_than_four());
  thrust::find_if_not(data, data+3, greater_than_four());
  thrust::find_if_not(device_data.begin(), device_data.end(),  greater_than_four());
  thrust::find_if_not(host_data.begin(), host_data.end(),  greater_than_four());
  thrust::find_if_not(thrust::host, data, data+3, greater_than_four());
  thrust::find_if_not(thrust::device, device_data.begin(), device_data.end(),  greater_than_four());
  thrust::find_if_not(thrust::host, host_data.begin(), host_data.end(),  greater_than_four());
}

void mismatch_test() {
  const int N = 4;
  int A[N] = {0, 5, 3, 7};
  int B[N] = {0, 5, 8, 7};

  thrust::host_vector<int> VA(A, A + N);
  thrust::host_vector<int> VB(B, B + N);
  thrust::device_vector<int> d_VA(A, A + N);
  thrust::device_vector<int> d_VB(B, B + N);

  // CHECK:  oneapi::dpl::mismatch(oneapi::dpl::execution::seq, VA.begin(), VA.end(), VB.begin());
  // CHECK-NEXT:  oneapi::dpl::mismatch(oneapi::dpl::execution::seq, VA.begin(), VA.end(), VB.begin());
  // CHECK-NEXT:  oneapi::dpl::mismatch(oneapi::dpl::execution::seq, VA.begin(), VA.end(), VB.begin(), oneapi::dpl::equal_to<int>());
  // CHECK-NEXT:  oneapi::dpl::mismatch(oneapi::dpl::execution::seq, VA.begin(), VA.end(), VB.begin(), oneapi::dpl::equal_to<int>());
  // CHECK-NEXT:  oneapi::dpl::mismatch(oneapi::dpl::execution::make_device_policy(q_ct1), d_VA.begin(), d_VA.end(), d_VB.begin());
  // CHECK-NEXT:  oneapi::dpl::mismatch(oneapi::dpl::execution::make_device_policy(q_ct1), d_VA.begin(), d_VA.end(), d_VB.begin());
  // CHECK-NEXT:  oneapi::dpl::mismatch(oneapi::dpl::execution::make_device_policy(q_ct1), d_VA.begin(), d_VA.end(), d_VB.begin(), oneapi::dpl::equal_to<int>());
  // CHECK-NEXT:  oneapi::dpl::mismatch(oneapi::dpl::execution::make_device_policy(q_ct1), d_VA.begin(), d_VA.end(), d_VB.begin(), oneapi::dpl::equal_to<int>());
  // CHECK-NEXT:  oneapi::dpl::mismatch(oneapi::dpl::execution::seq, A, A+N, B);
  // CHECK-NEXT:  oneapi::dpl::mismatch(oneapi::dpl::execution::seq, A, A+N, B);
  // CHECK-NEXT:  oneapi::dpl::mismatch(oneapi::dpl::execution::seq, A, A+N, B, oneapi::dpl::equal_to<int>());
  // CHECK-NEXT:  oneapi::dpl::mismatch(oneapi::dpl::execution::seq, A, A+N, B, oneapi::dpl::equal_to<int>());
  thrust::mismatch(thrust::host, VA.begin(), VA.end(), VB.begin());
  thrust::mismatch(VA.begin(), VA.end(), VB.begin());
  thrust::mismatch(thrust::host, VA.begin(), VA.end(), VB.begin(), thrust::equal_to<int>());
  thrust::mismatch(VA.begin(), VA.end(), VB.begin(), thrust::equal_to<int>());
  thrust::mismatch(thrust::device, d_VA.begin(), d_VA.end(), d_VB.begin());
  thrust::mismatch(d_VA.begin(), d_VA.end(), d_VB.begin());
  thrust::mismatch(thrust::device, d_VA.begin(), d_VA.end(), d_VB.begin(), thrust::equal_to<int>());
  thrust::mismatch( d_VA.begin(), d_VA.end(), d_VB.begin(), thrust::equal_to<int>());
  thrust::mismatch(thrust::host, A, A+N, B);
  thrust::mismatch( A, A+N, B);
  thrust::mismatch(thrust::host, A, A+N, B, thrust::equal_to<int>());
  thrust::mismatch( A, A+N, B, thrust::equal_to<int>());
}

void replace_copy_test() {
  const int N = 4;
  int data[] = {1, 2, 3, 1};
  thrust::device_vector<int> d_data(data, data +N);
  thrust::device_vector<int> d_result(4);
  thrust::host_vector<int> h_data(data, data +N);
  thrust::host_vector<int> h_result(4);
  int result[N];

  // CHECK:  oneapi::dpl::replace_copy(oneapi::dpl::execution::make_device_policy(q_ct1), d_data.begin(), d_data.end(), d_result.begin(), 1, 99);
  // CHECK-NEXT:  oneapi::dpl::replace_copy(oneapi::dpl::execution::seq, h_data.begin(), h_data.end(), h_result.begin(), 1, 99);
  // CHECK-NEXT:  oneapi::dpl::replace_copy(oneapi::dpl::execution::seq, data, data + N, result, 1, 99);
  // CHECK-NEXT:  oneapi::dpl::replace_copy(oneapi::dpl::execution::make_device_policy(q_ct1), d_data.begin(), d_data.end(), d_result.begin(), 1, 99);
  // CHECK-NEXT:  oneapi::dpl::replace_copy(oneapi::dpl::execution::seq, h_data.begin(), h_data.end(), h_result.begin(), 1, 99);
  // CHECK-NEXT:  oneapi::dpl::replace_copy(oneapi::dpl::execution::seq, data, data + N, result, 1, 99);
  thrust::replace_copy(thrust::device, d_data.begin(), d_data.end(), d_result.begin(), 1, 99);
  thrust::replace_copy(thrust::host, h_data.begin(), h_data.end(), h_result.begin(), 1, 99);
  thrust::replace_copy(thrust::host, data, data + N, result, 1, 99);
  thrust::replace_copy(d_data.begin(), d_data.end(), d_result.begin(), 1, 99);
  thrust::replace_copy(h_data.begin(), h_data.end(), h_result.begin(), 1, 99);
  thrust::replace_copy(data, data + N, result, 1, 99);
}


void reverse() {
  const int N = 6;
  int data[N] = {0, 1, 2, 3, 4, 5};
  thrust::device_vector<int> device_data(data, data + N);
  thrust::host_vector<int> host_data(data, data + N);

  // CHECK:  oneapi::dpl::reverse(oneapi::dpl::execution::make_device_policy(q_ct1), device_data.begin(), device_data.end());
  // CHECK-NEXT:  oneapi::dpl::reverse(oneapi::dpl::execution::seq, host_data.begin(), host_data.end());
  // CHECK-NEXT:  oneapi::dpl::reverse(oneapi::dpl::execution::seq, data, data + N);
  // CHECK-NEXT:  oneapi::dpl::reverse(oneapi::dpl::execution::make_device_policy(q_ct1), device_data.begin(), device_data.end());
  // CHECK-NEXT:  oneapi::dpl::reverse(oneapi::dpl::execution::seq, host_data.begin(), host_data.end());
  // CHECK-NEXT:  oneapi::dpl::reverse(oneapi::dpl::execution::seq, data, data + N);
  thrust::reverse(thrust::device, device_data.begin(), device_data.end());
  thrust::reverse(thrust::host, host_data.begin(), host_data.end());
  thrust::reverse(thrust::host, data, data + N);
  thrust::reverse(device_data.begin(), device_data.end());
  thrust::reverse(host_data.begin(), host_data.end());
  thrust::reverse(data, data + N);
}

void equal_range() {
  int data[] = {0, 2, 5, 7, 8};
  const int N = 5;
  thrust::host_vector<int> host_vec(data, data + N);
  thrust::device_vector<int> device_vec(data, data + N);

  // CHECK:  dpct::equal_range(oneapi::dpl::execution::make_device_policy(q_ct1), device_vec.begin(), device_vec.end(), 0); 
  // CHECK-NEXT:  dpct::equal_range(oneapi::dpl::execution::make_device_policy(q_ct1), device_vec.begin(), device_vec.end(), 0);
  // CHECK-NEXT:  dpct::equal_range(oneapi::dpl::execution::make_device_policy(q_ct1), device_vec.begin(), device_vec.end(), 0, oneapi::dpl::less<int>());
  // CHECK-NEXT:  dpct::equal_range(oneapi::dpl::execution::make_device_policy(q_ct1), device_vec.begin(), device_vec.end(), 0, oneapi::dpl::less<int>());
  // CHECK-NEXT:  dpct::equal_range(oneapi::dpl::execution::seq, host_vec.begin(), host_vec.end(), 0); 
  // CHECK-NEXT:  dpct::equal_range(oneapi::dpl::execution::seq, host_vec.begin(), host_vec.end(), 0);
  // CHECK-NEXT:  dpct::equal_range(oneapi::dpl::execution::seq, host_vec.begin(), host_vec.end(), 0, oneapi::dpl::less<int>());
  // CHECK-NEXT:  dpct::equal_range(oneapi::dpl::execution::seq, host_vec.begin(), host_vec.end(), 0, oneapi::dpl::less<int>());
  // CHECK-NEXT:  dpct::equal_range(oneapi::dpl::execution::seq, data, data + N, 0); 
  // CHECK-NEXT:  dpct::equal_range(oneapi::dpl::execution::seq, data, data + N, 0);
  // CHECK-NEXT:  dpct::equal_range(oneapi::dpl::execution::seq, data, data + N, 0, oneapi::dpl::less<int>()); 
  // CHECK-NEXT:  dpct::equal_range(oneapi::dpl::execution::seq, data, data + N, 0, oneapi::dpl::less<int>());
  thrust::equal_range(thrust::device, device_vec.begin(), device_vec.end(), 0); 
  thrust::equal_range(device_vec.begin(), device_vec.end(), 0);
  thrust::equal_range(thrust::device, device_vec.begin(), device_vec.end(), 0,  thrust::less<int>());
  thrust::equal_range(device_vec.begin(), device_vec.end(), 0,  thrust::less<int>());
  thrust::equal_range(thrust::host, host_vec.begin(), host_vec.end(), 0); 
  thrust::equal_range(host_vec.begin(), host_vec.end(), 0);
  thrust::equal_range(thrust::host, host_vec.begin(), host_vec.end(), 0,  thrust::less<int>());
  thrust::equal_range(host_vec.begin(), host_vec.end(), 0,  thrust::less<int>());
  thrust::equal_range(thrust::host, data, data + N, 0); 
  thrust::equal_range(data, data + N, 0);
  thrust::equal_range(thrust::host, data, data + N, 0, thrust::less<int>()); 
  thrust::equal_range(data, data + N, 0, thrust::less<int>());
}

void transform_inclusive_scan() {
  const int N = 6;
  int data[6] = {1, 0, 2, 2, 1, 3};
  thrust::negate<int> unary_op;
  thrust::plus<int> binary_op;
  thrust::host_vector<int> h_vec_data(data, data + N);
  thrust::device_vector<int> d_vec_data(data, data + N);

  // CHECK:  oneapi::dpl::transform_inclusive_scan(oneapi::dpl::execution::seq, data, data + N, data, binary_op, unary_op);
  // CHECK-NEXT:  oneapi::dpl::transform_inclusive_scan(oneapi::dpl::execution::seq, h_vec_data.begin(), h_vec_data.end(), h_vec_data.begin(), binary_op, unary_op);
  // CHECK-NEXT:  oneapi::dpl::transform_inclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), d_vec_data.begin(), d_vec_data.end(), d_vec_data.begin(), binary_op, unary_op);
  // CHECK-NEXT:  oneapi::dpl::transform_inclusive_scan(oneapi::dpl::execution::seq, data, data + N, data, binary_op, unary_op);
  // CHECK-NEXT:  oneapi::dpl::transform_inclusive_scan(oneapi::dpl::execution::seq, h_vec_data.begin(), h_vec_data.end(), h_vec_data.begin(), binary_op, unary_op);
  // CHECK-NEXT:  oneapi::dpl::transform_inclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), d_vec_data.begin(), d_vec_data.end(), d_vec_data.begin(), binary_op, unary_op);
  thrust::transform_inclusive_scan(data, data + N, data, unary_op, binary_op);
  thrust::transform_inclusive_scan(h_vec_data.begin(), h_vec_data.end(), h_vec_data.begin(), unary_op, binary_op);
  thrust::transform_inclusive_scan(d_vec_data.begin(), d_vec_data.end(), d_vec_data.begin(), unary_op, binary_op);
  thrust::transform_inclusive_scan(thrust::host, data, data + N, data, unary_op, binary_op);
  thrust::transform_inclusive_scan(thrust::host, h_vec_data.begin(), h_vec_data.end(), h_vec_data.begin(), unary_op, binary_op);
  thrust::transform_inclusive_scan(thrust::device, d_vec_data.begin(), d_vec_data.end(), d_vec_data.begin(), unary_op, binary_op);
}
