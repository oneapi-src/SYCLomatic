// UNSUPPORTED: system-windows
// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8
// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::adjacent_difference --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_adjacent_difference
// thrust_adjacent_difference: CUDA API:
// thrust_adjacent_difference-NEXT:   float *host_ptr_A;
// thrust_adjacent_difference-NEXT:   float *host_ptr_R;
// thrust_adjacent_difference-NEXT:   float *device_ptr_A;
// thrust_adjacent_difference-NEXT:   float *device_ptr_R;
// thrust_adjacent_difference-NEXT:   /*1*/ thrust::adjacent_difference(thrust::device, device_ptr_A, device_ptr_A+10, device_ptr_R);
// thrust_adjacent_difference-NEXT:   /*2*/ thrust::adjacent_difference(host_ptr_A, host_ptr_A+10, host_ptr_R, thrust::minus<float>());
// thrust_adjacent_difference-NEXT:   /*3*/ thrust::adjacent_difference(thrust::device, device_ptr_A, device_ptr_A+10, device_ptr_R, thrust::minus<float>());
// thrust_adjacent_difference-NEXT:   /*4*/ thrust::adjacent_difference(host_ptr_A, host_ptr_A+10, host_ptr_R);
// thrust_adjacent_difference-NEXT: Is migrated to:
// thrust_adjacent_difference-NEXT:   float *host_ptr_A;
// thrust_adjacent_difference-NEXT:   float *host_ptr_R;
// thrust_adjacent_difference-NEXT:   float *device_ptr_A;
// thrust_adjacent_difference-NEXT:   float *device_ptr_R;
// thrust_adjacent_difference-NEXT:  /*1*/ oneapi::dpl::adjacent_difference(oneapi::dpl::execution::make_device_policy(q_ct1), device_ptr_A, device_ptr_A+10, device_ptr_R);
// thrust_adjacent_difference-NEXT:  /*2*/ oneapi::dpl::adjacent_difference(oneapi::dpl::execution::seq, host_ptr_A, host_ptr_A+10, host_ptr_R, std::minus<float>());
// thrust_adjacent_difference-NEXT:  /*3*/ oneapi::dpl::adjacent_difference(oneapi::dpl::execution::make_device_policy(q_ct1), device_ptr_A, device_ptr_A+10, device_ptr_R, std::minus<float>());
// thrust_adjacent_difference-NEXT:  /*4*/ oneapi::dpl::adjacent_difference(oneapi::dpl::execution::seq, host_ptr_A, host_ptr_A+10, host_ptr_R);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::any_of --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_any_of
// thrust_any_of: CUDA API:
// thrust_any_of-NEXT:  struct greater_than_zero {
// thrust_any_of-NEXT:    __host__ __device__ bool operator()(int x) const { return x > 0; }
// thrust_any_of-NEXT:  };
// thrust_any_of-NEXT:  greater_than_zero pred;
// thrust_any_of-NEXT:  thrust::device_vector<int> A(4);
// thrust_any_of-NEXT:  thrust::device_vector<int> B(4);
// thrust_any_of-NEXT:  /*1*/ thrust::any_of(A.begin(), A.end(), pred);
// thrust_any_of-NEXT:  /*2*/ thrust::any_of(thrust::device, B.begin(), B.end(), pred);
// thrust_any_of-NEXT:Is migrated to:
// thrust_any_of-NEXT:  struct greater_than_zero {
// thrust_any_of-NEXT:    bool operator()(int x) const { return x > 0; }
// thrust_any_of-NEXT:  };
// thrust_any_of-NEXT:  greater_than_zero pred;
// thrust_any_of-NEXT:  dpct::device_vector<int> A(4);
// thrust_any_of-NEXT:  dpct::device_vector<int> B(4);
// thrust_any_of-NEXT:  /*1*/ oneapi::dpl::any_of(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), pred);
// thrust_any_of-NEXT:  /*2*/ oneapi::dpl::any_of(oneapi::dpl::execution::make_device_policy(q_ct1), B.begin(), B.end(), pred);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::binary_search --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_binary_search
// thrust_binary_search: CUDA API:
// thrust_binary_search-NEXT:  std::vector<int> v, v2, v3, v4;
// thrust_binary_search-NEXT:  auto bp = [](int x, int y) -> bool { return x < y; };
// thrust_binary_search-NEXT:  /*1*/ thrust::binary_search(thrust::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
// thrust_binary_search-NEXT:  /*2*/ thrust::binary_search(v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
// thrust_binary_search-NEXT:  /*3*/ thrust::binary_search(thrust::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
// thrust_binary_search-NEXT:  /*4*/ thrust::binary_search(v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
// thrust_binary_search-NEXT:Is migrated to:
// thrust_binary_search-NEXT:  std::vector<int> v, v2, v3, v4;
// thrust_binary_search-NEXT:  auto bp = [](int x, int y) -> bool { return x < y; };
// thrust_binary_search-NEXT:  /*1*/ oneapi::dpl::binary_search(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
// thrust_binary_search-NEXT:  /*2*/ oneapi::dpl::binary_search(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
// thrust_binary_search-NEXT:  /*3*/ oneapi::dpl::binary_search(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
// thrust_binary_search-NEXT:  /*4*/ oneapi::dpl::binary_search(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::copy_if --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_copy_if
// thrust_copy_if: CUDA API:
// thrust_copy_if-NEXT:  auto is_even = [] __host__ __device__(int v) { return (v % 2) == 0; };
// thrust_copy_if-NEXT:  const int N = 4;
// thrust_copy_if-NEXT:  int vec_host_in[N] = {-1, 0, 1, 2};
// thrust_copy_if-NEXT:  int vec_host_out[N];
// thrust_copy_if-NEXT:  int *vec_devic_in;
// thrust_copy_if-NEXT:  int *vec_devic_out;
// thrust_copy_if-NEXT:  thrust::device_vector<int> dVecIn(vec_host_in, vec_host_in + N);
// thrust_copy_if-NEXT:  thrust::device_vector<int> dVecOut(N);
// thrust_copy_if-NEXT:  /*1*/ thrust::copy_if(vec_host_in, vec_host_in + N, vec_host_out, is_even);
// thrust_copy_if-NEXT:  /*2*/ thrust::copy_if(thrust::device, vec_devic_in, vec_devic_in + N, vec_devic_out, is_even);
// thrust_copy_if-NEXT:  /*3*/ thrust::copy_if(thrust::host, vec_host_in, vec_host_in + N,vec_host_out, is_even);
// thrust_copy_if-NEXT:  /*4*/ thrust::copy_if(dVecIn.begin(), dVecIn.end(), dVecOut.begin(), is_even);
// thrust_copy_if-NEXT:Is migrated to:
// thrust_copy_if-NEXT:  auto is_even = [] (int v) { return (v % 2) == 0; };
// thrust_copy_if-NEXT:  const int N = 4;
// thrust_copy_if-NEXT:  int vec_host_in[N] = {-1, 0, 1, 2};
// thrust_copy_if-NEXT:  int vec_host_out[N];
// thrust_copy_if-NEXT:  int *vec_devic_in;
// thrust_copy_if-NEXT:  int *vec_devic_out;
// thrust_copy_if-NEXT:  dpct::device_vector<int> dVecIn(vec_host_in, vec_host_in + N);
// thrust_copy_if-NEXT:  dpct::device_vector<int> dVecOut(N);
// thrust_copy_if-NEXT:  /*1*/ std::copy_if(oneapi::dpl::execution::seq, vec_host_in, vec_host_in + N, vec_host_out, is_even);
// thrust_copy_if-NEXT:  /*2*/ std::copy_if(oneapi::dpl::execution::make_device_policy(q_ct1), vec_devic_in, vec_devic_in + N, vec_devic_out, is_even);
// thrust_copy_if-NEXT:  /*3*/ std::copy_if(oneapi::dpl::execution::seq, vec_host_in, vec_host_in + N, vec_host_out, is_even);
// thrust_copy_if-NEXT:  /*4*/ std::copy_if(oneapi::dpl::execution::make_device_policy(q_ct1), dVecIn.begin(), dVecIn.end(), dVecOut.begin(), is_even);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::count_if --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_count_if
// thrust_count_if: CUDA API:
// thrust_count_if-NEXT:  std::vector<thrust::device_vector<int>> d(10);
// thrust_count_if-NEXT:  auto t = thrust::make_counting_iterator(0);
// thrust_count_if-NEXT:  int ret = thrust::count_if(t, t + 10, [=] __device__(int idx) { return true;});
// thrust_count_if-NEXT:Is migrated to:
// thrust_count_if-NEXT:  std::vector<dpct::device_vector<int>> d(10);
// thrust_count_if-NEXT:  auto t = dpct::make_counting_iterator(0);
// thrust_count_if-NEXT:  int ret = std::count_if(oneapi::dpl::execution::seq, t, t + 10, [=] (int idx) { return true;});

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::exclusive_scan --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_exclusive_scan
// thrust_exclusive_scan: CUDA API:
// thrust_exclusive_scan-NEXT:  std::vector<int> v, v2, v3, v4;
// thrust_exclusive_scan-NEXT:  thrust::maximum<int> binary_op;
// thrust_exclusive_scan-NEXT:  thrust::device_vector<int> tv, tv2, tv3, tv4;
// thrust_exclusive_scan-NEXT:  /*1*/ thrust::exclusive_scan(thrust::device, tv.begin(), tv.end(), tv2.begin());
// thrust_exclusive_scan-NEXT:  /*2*/ thrust::exclusive_scan(tv.begin(), tv.end(), tv2.begin());
// thrust_exclusive_scan-NEXT:  /*3*/ thrust::exclusive_scan(thrust::device, tv.begin(), tv.end(), tv2.begin(), 4);
// thrust_exclusive_scan-NEXT:  /*4*/ thrust::exclusive_scan(tv.begin(), tv.end(), tv2.begin(), 4);
// thrust_exclusive_scan-NEXT:  /*5*/ thrust::exclusive_scan(thrust::device, tv.begin(), tv.end(), tv2.begin(), 1, binary_op);
// thrust_exclusive_scan-NEXT:  /*6*/ thrust::exclusive_scan(tv.begin(), tv.end(), tv2.begin(), 1, binary_op);
// thrust_exclusive_scan-NEXT:Is migrated to:
// thrust_exclusive_scan-NEXT:  std::vector<int> v, v2, v3, v4;
// thrust_exclusive_scan-NEXT:  oneapi::dpl::maximum<int> binary_op;
// thrust_exclusive_scan-NEXT:  dpct::device_vector<int> tv, tv2, tv3, tv4;
// thrust_exclusive_scan-NEXT:  /*1*/ std::exclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), tv.begin(), tv.end(), tv2.begin(), (decltype(tv2.begin())::value_type)0);
// thrust_exclusive_scan-NEXT:  /*2*/ std::exclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), tv.begin(), tv.end(), tv2.begin(), (decltype(tv2.begin())::value_type)0);
// thrust_exclusive_scan-NEXT:  /*3*/ std::exclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), tv.begin(), tv.end(), tv2.begin(), (decltype(tv2.begin())::value_type)4);
// thrust_exclusive_scan-NEXT:  /*4*/ std::exclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), tv.begin(), tv.end(), tv2.begin(), (decltype(tv2.begin())::value_type)4);
// thrust_exclusive_scan-NEXT:  /*5*/ std::exclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), tv.begin(), tv.end(), tv2.begin(), (decltype(tv2.begin())::value_type)1, binary_op);
// thrust_exclusive_scan-NEXT:  /*6*/ std::exclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), tv.begin(), tv.end(), tv2.begin(), (decltype(tv2.begin())::value_type)1, binary_op);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::exclusive_scan_by_key --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_exclusive_scan_by_key
// thrust_exclusive_scan_by_key: CUDA API:
// thrust_exclusive_scan_by_key-NEXT:  std::vector<int> v, v2, v3, v4;
// thrust_exclusive_scan_by_key-NEXT:  auto bp = [](int x, int y) -> bool { return x < y; };
// thrust_exclusive_scan_by_key-NEXT:  thrust::device_vector<int> tv, tv2, tv3, tv4;
// thrust_exclusive_scan_by_key-NEXT:  /*1*/ thrust::exclusive_scan_by_key(thrust::seq, v.begin(), v.end(), v2.begin(), v3.begin());
// thrust_exclusive_scan_by_key-NEXT:  /*2*/ thrust::exclusive_scan_by_key(v.begin(), v.end(), v2.begin(), v3.begin());
// thrust_exclusive_scan_by_key-NEXT:  /*3*/ thrust::exclusive_scan_by_key(thrust::seq, v.begin(), v.end(), v2.begin(), v3.begin(), 1);
// thrust_exclusive_scan_by_key-NEXT:  /*4*/ thrust::exclusive_scan_by_key(v.begin(), v.end(), v2.begin(), v3.begin(), 1);
// thrust_exclusive_scan_by_key-NEXT:  /*5*/ thrust::exclusive_scan_by_key(thrust::seq, v.begin(), v.end(), v2.begin(), v3.begin(), 1, bp);
// thrust_exclusive_scan_by_key-NEXT:  /*6*/ thrust::exclusive_scan_by_key(v.begin(), v.end(), v2.begin(), v3.begin(), 1, bp);
// thrust_exclusive_scan_by_key-NEXT:Is migrated to:
// thrust_exclusive_scan_by_key-NEXT:  std::vector<int> v, v2, v3, v4;
// thrust_exclusive_scan_by_key-NEXT:  auto bp = [](int x, int y) -> bool { return x < y; };
// thrust_exclusive_scan_by_key-NEXT:  dpct::device_vector<int> tv, tv2, tv3, tv4;
// thrust_exclusive_scan_by_key-NEXT:  /*1*/ oneapi::dpl::exclusive_scan_by_segment(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v3.begin());
// thrust_exclusive_scan_by_key-NEXT:  /*2*/ oneapi::dpl::exclusive_scan_by_segment(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v3.begin());
// thrust_exclusive_scan_by_key-NEXT:  /*3*/ oneapi::dpl::exclusive_scan_by_segment(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v3.begin(), 1);
// thrust_exclusive_scan_by_key-NEXT:  /*4*/ oneapi::dpl::exclusive_scan_by_segment(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v3.begin(), 1);
// thrust_exclusive_scan_by_key-NEXT:  /*5*/ oneapi::dpl::exclusive_scan_by_segment(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v3.begin(), 1, bp);
// thrust_exclusive_scan_by_key-NEXT:  /*6*/ oneapi::dpl::exclusive_scan_by_segment(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v3.begin(), 1, bp);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::fill --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_fill
// thrust_fill: CUDA API:
// thrust_fill-NEXT:  float *_de = NULL;
// thrust_fill-NEXT:  float fill_value = 0.0;
// thrust_fill-NEXT:  thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(static_cast<float *>(&_de[0]));
// thrust_fill-NEXT:  thrust::fill(dev_ptr, dev_ptr + 10, fill_value);
// thrust_fill-NEXT:Is migrated to:
// thrust_fill-NEXT:  float *_de = NULL;
// thrust_fill-NEXT:  float fill_value = 0.0;
// thrust_fill-NEXT:  dpct::device_pointer<float> dev_ptr = dpct::get_device_pointer(static_cast<float *>(&_de[0]));
// thrust_fill-NEXT:  std::fill(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), dev_ptr, dev_ptr + 10, fill_value);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::fill_n --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_fill_n
// thrust_fill_n: CUDA API:
// thrust_fill_n-NEXT:  float *_de = NULL;
// thrust_fill_n-NEXT:  float fill_value = 0.0;
// thrust_fill_n-NEXT:  thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(static_cast<float *>(&_de[0]));
// thrust_fill_n-NEXT:  thrust::fill_n(dev_ptr, 10, fill_value);
// thrust_fill_n-NEXT:Is migrated to:
// thrust_fill_n-NEXT:  float *_de = NULL;
// thrust_fill_n-NEXT:  float fill_value = 0.0;
// thrust_fill_n-NEXT:  dpct::device_pointer<float> dev_ptr = dpct::get_device_pointer(static_cast<float *>(&_de[0]));
// thrust_fill_n-NEXT:  std::fill_n(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), dev_ptr, 10, fill_value);


// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::find --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_find
// thrust_find: CUDA API:
// thrust_find-NEXT:  thrust::host_vector<int> h;
// thrust_find-NEXT:  thrust::device_vector<int> d;
// thrust_find-NEXT:  /*1*/ thrust::find(thrust::seq, h.begin(), h.end(), 1);
// thrust_find-NEXT:  /*2*/ thrust::find(h.begin(), h.end(), 1);
// thrust_find-NEXT:  /*3*/ thrust::find(d.begin(), d.end(), 1);
// thrust_find-NEXT:Is migrated to:
// thrust_find-NEXT:  std::vector<int> h;
// thrust_find-NEXT:  dpct::device_vector<int> d;
// thrust_find-NEXT:  /*1*/ oneapi::dpl::find(oneapi::dpl::execution::seq, h.begin(), h.end(), 1);
// thrust_find-NEXT:  /*2*/ oneapi::dpl::find(oneapi::dpl::execution::seq, h.begin(), h.end(), 1);
// thrust_find-NEXT:  /*3*/ oneapi::dpl::find(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), d.begin(), d.end(), 1);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::for_each --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_for_each
// thrust_for_each: CUDA API:
// thrust_for_each-NEXT:   struct add_functor {
// thrust_for_each-NEXT:     __host__ __device__ void operator()(int &x) { x++; }
// thrust_for_each-NEXT:   };
// thrust_for_each-NEXT:   auto loop_body = [=] __device__ __host__(int ind) -> void {};
// thrust_for_each-NEXT:   thrust::device_vector<int> t;
// thrust_for_each-NEXT:   /*1*/ thrust::for_each(t.begin(), t.end(), add_functor());
// thrust_for_each-NEXT:   /*2*/ thrust::for_each(thrust::cuda::par_nosync, t.begin(), t.end(), loop_body );
// thrust_for_each-NEXT: Is migrated to:
// thrust_for_each-NEXT:   struct add_functor {
// thrust_for_each-NEXT:     void operator()(int &x) const { x++; }
// thrust_for_each-NEXT:   };
// thrust_for_each-NEXT:   auto loop_body = [=] (int ind) -> void {};
// thrust_for_each-NEXT:   dpct::device_vector<int> t;
// thrust_for_each-NEXT:   /*1*/ std::for_each(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), t.begin(), t.end(), add_functor());
// thrust_for_each-NEXT:   /*2*/ std::for_each(par_nosync, t.begin(), t.end(), loop_body);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::gather --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_gather
// thrust_gather: CUDA API:
// thrust_gather-NEXT:   int values[10] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
// thrust_gather-NEXT:   thrust::host_vector<int> h_values(values, values + 10);
// thrust_gather-NEXT:   int map[10] = {0, 2, 4, 6, 8, 1, 3, 5, 7, 9};
// thrust_gather-NEXT:   thrust::host_vector<int> h_map(map, map + 10);
// thrust_gather-NEXT:   thrust::host_vector<int> h_output(10);
// thrust_gather-NEXT:   thrust::device_vector<int> d_values(values, values + 10);
// thrust_gather-NEXT:   thrust::device_vector<int> d_map(map, map + 10);
// thrust_gather-NEXT:   thrust::device_vector<int> d_output(10);
// thrust_gather-NEXT:   /*1*/ thrust::gather(d_map.begin(), d_map.end(), d_values.begin(),d_output.begin());
// thrust_gather-NEXT:   /*2*/ thrust::gather(thrust::device, d_map.begin(), d_map.end(),d_values.begin(), d_output.begin());
// thrust_gather-NEXT:   /*3*/ thrust::gather(thrust::seq, h_map.begin(), h_map.end(),h_values.begin(), h_output.begin());
// thrust_gather-NEXT:   /*4*/ thrust::gather(h_map.begin(), h_map.end(), h_values.begin(),h_output.begin());
// thrust_gather-NEXT: Is migrated to:
// thrust_gather-NEXT:   int values[10] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
// thrust_gather-NEXT:   std::vector<int> h_values(values, values + 10);
// thrust_gather-NEXT:   int map[10] = {0, 2, 4, 6, 8, 1, 3, 5, 7, 9};
// thrust_gather-NEXT:   std::vector<int> h_map(map, map + 10);
// thrust_gather-NEXT:   std::vector<int> h_output(10);
// thrust_gather-NEXT:   dpct::device_vector<int> d_values(values, values + 10);
// thrust_gather-NEXT:   dpct::device_vector<int> d_map(map, map + 10);
// thrust_gather-NEXT:   dpct::device_vector<int> d_output(10);
// thrust_gather-NEXT:   /*1*/ dpct::gather(oneapi::dpl::execution::make_device_policy(q_ct1), d_map.begin(), d_map.end(), d_values.begin(), d_output.begin());
// thrust_gather-NEXT:   /*2*/ dpct::gather(oneapi::dpl::execution::make_device_policy(q_ct1), d_map.begin(), d_map.end(), d_values.begin(), d_output.begin());
// thrust_gather-NEXT:   /*3*/ dpct::gather(oneapi::dpl::execution::seq, h_map.begin(), h_map.end(), h_values.begin(), h_output.begin());
// thrust_gather-NEXT:   /*4*/ dpct::gather(oneapi::dpl::execution::seq, h_map.begin(), h_map.end(), h_values.begin(), h_output.begin());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::generate_n --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_generate_n
// thrust_generate_n: CUDA API:
// thrust_generate_n-NEXT:   std::vector<int> v, v2, v3, v4;
// thrust_generate_n-NEXT:   auto gen = []() -> int { return 23; };
// thrust_generate_n-NEXT:   /*1*/ thrust::generate_n(thrust::seq, v.begin(), 23, gen);
// thrust_generate_n-NEXT:   /*2*/ thrust::generate_n(v.begin(), 23, gen);
// thrust_generate_n-NEXT: Is migrated to:
// thrust_generate_n-NEXT:   std::vector<int> v, v2, v3, v4;
// thrust_generate_n-NEXT:   auto gen = []() -> int { return 23; };
// thrust_generate_n-NEXT:   /*1*/ std::generate_n(oneapi::dpl::execution::seq, v.begin(), 23, gen);
// thrust_generate_n-NEXT:   /*2*/ std::generate_n(oneapi::dpl::execution::seq, v.begin(), 23, gen);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::generate --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_generate
// thrust_generate: CUDA API:
// thrust_generate-NEXT:   std::vector<int> v, v2, v3, v4;
// thrust_generate-NEXT:   auto gen = []() -> int { return 23; };
// thrust_generate-NEXT:   /*1*/ thrust::generate(thrust::seq, v.begin(), v.end(), gen);
// thrust_generate-NEXT:   /*2*/ thrust::generate(v.begin(), v.end(), gen);
// thrust_generate-NEXT: Is migrated to:
// thrust_generate-NEXT:   std::vector<int> v, v2, v3, v4;
// thrust_generate-NEXT:   auto gen = []() -> int { return 23; };
// thrust_generate-NEXT:   /*1*/ std::generate(oneapi::dpl::execution::seq, v.begin(), v.end(), gen);
// thrust_generate-NEXT:   /*2*/ std::generate(oneapi::dpl::execution::seq, v.begin(), v.end(), gen);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::inclusive_scan --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_inclusive_scan
// thrust_inclusive_scan: CUDA API:
// thrust_inclusive_scan-NEXT:   thrust::device_vector<int> A(4);
// thrust_inclusive_scan-NEXT:   std::vector<int> B(4);
// thrust_inclusive_scan-NEXT:   thrust::device_vector<int> R(4);
// thrust_inclusive_scan-NEXT:   std::vector<int> R2(4);
// thrust_inclusive_scan-NEXT:   /*1*/ thrust::inclusive_scan(B.begin(), B.end(), R2.begin(), thrust::minus<int>());
// thrust_inclusive_scan-NEXT:   /*2*/ thrust::inclusive_scan(thrust::device, A.begin(), A.end(), R.begin(),thrust::minus<int>());
// thrust_inclusive_scan-NEXT:   /*3*/ thrust::inclusive_scan(A.begin(), A.end(), R.begin());
// thrust_inclusive_scan-NEXT:   /*4*/ thrust::inclusive_scan(thrust::device, A.begin(), A.end(), R.begin());
// thrust_inclusive_scan-NEXT: Is migrated to:
// thrust_inclusive_scan-NEXT:   dpct::device_vector<int> A(4);
// thrust_inclusive_scan-NEXT:   std::vector<int> B(4);
// thrust_inclusive_scan-NEXT:   dpct::device_vector<int> R(4);
// thrust_inclusive_scan-NEXT:   std::vector<int> R2(4);
// thrust_inclusive_scan-NEXT:   /*1*/ oneapi::dpl::inclusive_scan(oneapi::dpl::execution::seq, B.begin(), B.end(), R2.begin(), std::minus<int>());
// thrust_inclusive_scan-NEXT:   /*2*/ oneapi::dpl::inclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), R.begin(), std::minus<int>());
// thrust_inclusive_scan-NEXT:   /*3*/ oneapi::dpl::inclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), R.begin());
// thrust_inclusive_scan-NEXT:   /*4*/ oneapi::dpl::inclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), R.begin());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::inclusive_scan_by_key --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_inclusive_scan_by_key
// thrust_inclusive_scan_by_key: CUDA API:
// thrust_inclusive_scan_by_key-NEXT:   std::vector<int> v, v2, v3, v4;
// thrust_inclusive_scan_by_key-NEXT:   auto bp = [](int x, int y) -> bool { return x < y; };
// thrust_inclusive_scan_by_key-NEXT:   auto bo = [](int x, int y) -> int { return x + y; };
// thrust_inclusive_scan_by_key-NEXT:   thrust::device_vector<int> tv, tv2, tv3, tv4;
// thrust_inclusive_scan_by_key-NEXT:   /*1*/ thrust::inclusive_scan_by_key(v.begin(), v.end(), v2.begin(), v3.begin());
// thrust_inclusive_scan_by_key-NEXT:   /*2*/ thrust::inclusive_scan_by_key(thrust::seq, v.begin(), v.end(), v2.begin(), v3.begin(), bp);
// thrust_inclusive_scan_by_key-NEXT:   /*3*/ thrust::inclusive_scan_by_key(v.begin(), v.end(), v2.begin(), v3.begin(), bp);
// thrust_inclusive_scan_by_key-NEXT:   /*4*/ thrust::inclusive_scan_by_key(v.begin(), v.end(), v2.begin(), v3.begin(), bp, bo);
// thrust_inclusive_scan_by_key-NEXT: Is migrated to:
// thrust_inclusive_scan_by_key-NEXT:   std::vector<int> v, v2, v3, v4;
// thrust_inclusive_scan_by_key-NEXT:   auto bp = [](int x, int y) -> bool { return x < y; };
// thrust_inclusive_scan_by_key-NEXT:   auto bo = [](int x, int y) -> int { return x + y; };
// thrust_inclusive_scan_by_key-NEXT:   dpct::device_vector<int> tv, tv2, tv3, tv4;
// thrust_inclusive_scan_by_key-NEXT:   /*1*/ oneapi::dpl::inclusive_scan_by_segment(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v3.begin());
// thrust_inclusive_scan_by_key-NEXT:   /*2*/ oneapi::dpl::inclusive_scan_by_segment(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v3.begin(), bp);
// thrust_inclusive_scan_by_key-NEXT:   /*3*/ oneapi::dpl::inclusive_scan_by_segment(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v3.begin(), bp);
// thrust_inclusive_scan_by_key-NEXT:   /*4*/ oneapi::dpl::inclusive_scan_by_segment(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v3.begin(), bp, bo);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::inner_product --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_inner_product
// thrust_inner_product: CUDA API:
// thrust_inner_product-NEXT:   thrust::multiplies<int> bo1;
// thrust_inner_product-NEXT:   thrust::multiplies<int> bo2;
// thrust_inner_product-NEXT:   thrust::host_vector<int> h;
// thrust_inner_product-NEXT:   thrust::device_vector<int> d;
// thrust_inner_product-NEXT:   /*1*/ thrust::inner_product(thrust::host, h.begin(), h.end(), h.begin(), 1);
// thrust_inner_product-NEXT:   /*2*/ thrust::inner_product(thrust::device, d.begin(), d.end(), d.begin(), 1, bo1, bo2);
// thrust_inner_product-NEXT:   /*3*/ thrust::inner_product(d.begin(), d.end(), d.begin(), 1);
// thrust_inner_product-NEXT:   /*4*/ thrust::inner_product(d.begin(), d.end(), d.begin(), 1, bo1, bo2);
// thrust_inner_product-NEXT: Is migrated to:
// thrust_inner_product-NEXT:   std::multiplies<int> bo1;
// thrust_inner_product-NEXT:   std::multiplies<int> bo2;
// thrust_inner_product-NEXT:   std::vector<int> h;
// thrust_inner_product-NEXT:   dpct::device_vector<int> d;
// thrust_inner_product-NEXT:   /*1*/ dpct::inner_product(oneapi::dpl::execution::seq, h.begin(), h.end(), h.begin(), 1);
// thrust_inner_product-NEXT:   /*2*/ dpct::inner_product(oneapi::dpl::execution::make_device_policy(q_ct1), d.begin(), d.end(), d.begin(), 1, bo1, bo2);
// thrust_inner_product-NEXT:   /*3*/ dpct::inner_product(oneapi::dpl::execution::make_device_policy(q_ct1), d.begin(), d.end(), d.begin(), 1);
// thrust_inner_product-NEXT:   /*4*/ dpct::inner_product(oneapi::dpl::execution::make_device_policy(q_ct1), d.begin(), d.end(), d.begin(), 1, bo1, bo2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::lower_bound --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_lower_bound
// thrust_lower_bound: CUDA API:
// thrust_lower_bound-NEXT:   std::vector<int> v, v2, v3, v4;
// thrust_lower_bound-NEXT:   thrust::device_vector<int> tv, tv2, tv3, tv4;
// thrust_lower_bound-NEXT:   auto bp = [](int x, int y) -> bool { return x < y; };
// thrust_lower_bound-NEXT:   /*1*/ thrust::lower_bound(thrust::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
// thrust_lower_bound-NEXT:   /*2*/ thrust::lower_bound(v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
// thrust_lower_bound-NEXT:   /*3*/ thrust::lower_bound(thrust::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
// thrust_lower_bound-NEXT:   /*4*/ thrust::lower_bound(v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
// thrust_lower_bound-NEXT: Is migrated to:
// thrust_lower_bound-NEXT:   std::vector<int> v, v2, v3, v4;
// thrust_lower_bound-NEXT:   dpct::device_vector<int> tv, tv2, tv3, tv4;
// thrust_lower_bound-NEXT:   auto bp = [](int x, int y) -> bool { return x < y; };
// thrust_lower_bound-NEXT:   /*1*/ oneapi::dpl::lower_bound(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
// thrust_lower_bound-NEXT:   /*2*/ oneapi::dpl::lower_bound(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
// thrust_lower_bound-NEXT:   /*3*/ oneapi::dpl::lower_bound(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
// thrust_lower_bound-NEXT:   /*4*/ oneapi::dpl::lower_bound(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::max_element --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_max_element
// thrust_max_element: CUDA API:
// thrust_max_element-NEXT:   thrust::host_vector<int> h_input(10);
// thrust_max_element-NEXT:   thrust::max_element(h_input.begin(), h_input.end());
// thrust_max_element-NEXT: Is migrated to:
// thrust_max_element-NEXT:   std::vector<int> h_input(10);
// thrust_max_element-NEXT:   std::max_element(oneapi::dpl::execution::seq, h_input.begin(), h_input.end());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::merge --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_merge
// thrust_merge: CUDA API:
// thrust_merge-NEXT:   std::vector<int> v, v2, v3, v4;
// thrust_merge-NEXT:   auto bp = [](int x, int y) -> bool { return x < y; };
// thrust_merge-NEXT:   thrust::device_vector<int> tv, tv2, tv3, tv4;
// thrust_merge-NEXT:   /*1*/ thrust::merge(thrust::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
// thrust_merge-NEXT:   /*2*/ thrust::merge(v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
// thrust_merge-NEXT:   /*3*/ thrust::merge(thrust::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
// thrust_merge-NEXT:   /*4*/ thrust::merge(v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
// thrust_merge-NEXT: Is migrated to:
// thrust_merge-NEXT:   std::vector<int> v, v2, v3, v4;
// thrust_merge-NEXT:   auto bp = [](int x, int y) -> bool { return x < y; };
// thrust_merge-NEXT:   dpct::device_vector<int> tv, tv2, tv3, tv4;
// thrust_merge-NEXT:   /*1*/ std::merge(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
// thrust_merge-NEXT:   /*2*/ std::merge(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
// thrust_merge-NEXT:   /*3*/ std::merge(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
// thrust_merge-NEXT:   /*4*/ std::merge(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::merge_by_key --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_merge_by_key
// thrust_merge_by_key: CUDA API:
// thrust_merge_by_key-NEXT:   thrust::host_vector<int> AH(4);
// thrust_merge_by_key-NEXT:   thrust::host_vector<int> BH(4);
// thrust_merge_by_key-NEXT:   thrust::host_vector<int> CH(4);
// thrust_merge_by_key-NEXT:   thrust::host_vector<int> DH(4);
// thrust_merge_by_key-NEXT:   thrust::host_vector<int> EH(8);
// thrust_merge_by_key-NEXT:   thrust::host_vector<int> FH(8);
// thrust_merge_by_key-NEXT:   /*1*/ thrust::merge_by_key(AH.begin(), AH.end(), BH.begin(), BH.end(), CH.begin(),DH.begin(), EH.begin(), FH.begin());
// thrust_merge_by_key-NEXT:   /*2*/ thrust::merge_by_key(AH.begin(), AH.end(), BH.begin(), BH.end(), CH.begin(),DH.begin(), EH.begin(), FH.begin(), thrust::greater<int>());
// thrust_merge_by_key-NEXT:   /*3*/ thrust::merge_by_key(thrust::host, AH.begin(), AH.end(), BH.begin(), BH.end(), CH.begin(), DH.begin(), EH.begin(), FH.begin());
// thrust_merge_by_key-NEXT:   /*4*/ thrust::merge_by_key(thrust::host, AH.begin(), AH.end(), BH.begin(), BH.end(), CH.begin(), DH.begin(), EH.begin(), FH.begin(), thrust::greater<int>());
// thrust_merge_by_key-NEXT: Is migrated to:
// thrust_merge_by_key-NEXT:   std::vector<int> AH(4);
// thrust_merge_by_key-NEXT:   std::vector<int> BH(4);
// thrust_merge_by_key-NEXT:   std::vector<int> CH(4);
// thrust_merge_by_key-NEXT:   std::vector<int> DH(4);
// thrust_merge_by_key-NEXT:   std::vector<int> EH(8);
// thrust_merge_by_key-NEXT:   std::vector<int> FH(8);
// thrust_merge_by_key-NEXT:   /*1*/ dpct::merge(oneapi::dpl::execution::seq, AH.begin(), AH.end(), BH.begin(), BH.end(), CH.begin(), DH.begin(), EH.begin(), FH.begin());
// thrust_merge_by_key-NEXT:   /*2*/ dpct::merge(oneapi::dpl::execution::seq, AH.begin(), AH.end(), BH.begin(), BH.end(), CH.begin(), DH.begin(), EH.begin(), FH.begin(), std::greater<int>());
// thrust_merge_by_key-NEXT:   /*3*/ dpct::merge(oneapi::dpl::execution::seq, AH.begin(), AH.end(), BH.begin(), BH.end(), CH.begin(), DH.begin(), EH.begin(), FH.begin());
// thrust_merge_by_key-NEXT:   /*4*/ dpct::merge(oneapi::dpl::execution::seq, AH.begin(), AH.end(), BH.begin(), BH.end(), CH.begin(), DH.begin(), EH.begin(), FH.begin(), std::greater<int>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::min_element --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_min_element
// thrust_min_element: CUDA API:
// thrust_min_element-NEXT:   thrust::host_vector<int> h_input(10);
// thrust_min_element-NEXT:   thrust::min_element(h_input.begin(), h_input.end());
// thrust_min_element-NEXT: Is migrated to:
// thrust_min_element-NEXT:   std::vector<int> h_input(10);
// thrust_min_element-NEXT:   std::min_element(oneapi::dpl::execution::seq, h_input.begin(), h_input.end());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::reduce --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_reduce
// thrust_reduce: CUDA API:
// thrust_reduce-NEXT:   int data[6] = {1, 0, 2, 2, 1, 3};
// thrust_reduce-NEXT:   thrust::device_vector<int> d_data(data, data + 6);
// thrust_reduce-NEXT:   int result;
// thrust_reduce-NEXT:   /*1*/ result = thrust::reduce(thrust::device, d_data.begin(), d_data.begin() + 6);
// thrust_reduce-NEXT:   /*2*/ result = thrust::reduce(thrust::device, d_data.begin(), d_data.begin() + 6, 1);
// thrust_reduce-NEXT:   /*3*/ result = thrust::reduce(d_data.begin(), d_data.begin() + 6, 1);
// thrust_reduce-NEXT:   /*4*/ result = thrust::reduce(thrust::device, d_data.begin(), d_data.begin() + 6, -1, thrust::maximum<int>());
// thrust_reduce-NEXT:   /*5*/ result = thrust::reduce(d_data.begin(), d_data.begin() + 6, -1, thrust::maximum<int>());
// thrust_reduce-NEXT: Is migrated to:
// thrust_reduce-NEXT:   int data[6] = {1, 0, 2, 2, 1, 3};
// thrust_reduce-NEXT:   dpct::device_vector<int> d_data(data, data + 6);
// thrust_reduce-NEXT:   int result;
// thrust_reduce-NEXT:   /*1*/ result = std::reduce(oneapi::dpl::execution::make_device_policy(q_ct1), d_data.begin(), d_data.begin() + 6);
// thrust_reduce-NEXT:   /*2*/ result = std::reduce(oneapi::dpl::execution::make_device_policy(q_ct1), d_data.begin(), d_data.begin() + 6, 1);
// thrust_reduce-NEXT:   /*3*/ result = std::reduce(oneapi::dpl::execution::make_device_policy(q_ct1), d_data.begin(), d_data.begin() + 6, 1);
// thrust_reduce-NEXT:   /*4*/ result = std::reduce(oneapi::dpl::execution::make_device_policy(q_ct1), d_data.begin(), d_data.begin() + 6, -1, oneapi::dpl::maximum<int>());
// thrust_reduce-NEXT:   /*5*/ result = std::reduce(oneapi::dpl::execution::make_device_policy(q_ct1), d_data.begin(), d_data.begin() + 6, -1, oneapi::dpl::maximum<int>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::reduce_by_key --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_reduce_by_key
// thrust_reduce_by_key: CUDA API:
// thrust_reduce_by_key-NEXT:   thrust::host_vector<int> h;
// thrust_reduce_by_key-NEXT:   thrust::device_vector<int> d;
// thrust_reduce_by_key-NEXT:   thrust::not_equal_to<int> bp;
// thrust_reduce_by_key-NEXT:   thrust::multiplies<int> bo1;
// thrust_reduce_by_key-NEXT:   /*1*/ thrust::reduce_by_key(thrust::host, h.begin(), h.end(), h.begin(), h.end(), h.begin(), bp, bo1);
// thrust_reduce_by_key-NEXT:   /*2*/thrust::reduce_by_key(thrust::device, d.begin(), d.end(), d.begin(), d.end(), d.begin(), bp);
// thrust_reduce_by_key-NEXT:   /*3*/thrust::reduce_by_key(d.begin(), d.end(), d.begin(), d.end(), d.begin(), bp, bo1);
// thrust_reduce_by_key-NEXT:   /*4*/thrust::reduce_by_key(thrust::host, h.begin(), h.end(), thrust::constant_iterator<int>(1), h.end(), h.begin());
// thrust_reduce_by_key-NEXT:   /*5*/thrust::reduce_by_key(d.begin(), d.end(), d.begin(), d.end(), d.begin(), bp);
// thrust_reduce_by_key-NEXT:   /*6*/thrust::reduce_by_key(d.begin(), d.end(), d.begin(), d.end(), d.begin());
// thrust_reduce_by_key-NEXT: Is migrated to:
// thrust_reduce_by_key-NEXT:   std::vector<int> h;
// thrust_reduce_by_key-NEXT:   dpct::device_vector<int> d;
// thrust_reduce_by_key-NEXT:   std::not_equal_to<int> bp;
// thrust_reduce_by_key-NEXT:   std::multiplies<int> bo1;
// thrust_reduce_by_key-NEXT:   /*1*/ oneapi::dpl::reduce_by_segment(oneapi::dpl::execution::seq, h.begin(), h.end(), h.begin(), h.end(), h.begin(), bp, bo1);
// thrust_reduce_by_key-NEXT:   /*2*/oneapi::dpl::reduce_by_segment(oneapi::dpl::execution::make_device_policy(q_ct1), d.begin(), d.end(), d.begin(), d.end(), d.begin(), bp);
// thrust_reduce_by_key-NEXT:   /*3*/oneapi::dpl::reduce_by_segment(oneapi::dpl::execution::make_device_policy(q_ct1), d.begin(), d.end(), d.begin(), d.end(), d.begin(), bp, bo1);
// thrust_reduce_by_key-NEXT:   /*4*/oneapi::dpl::reduce_by_segment(oneapi::dpl::execution::seq, h.begin(), h.end(), dpct::constant_iterator<int>(1), h.end(), h.begin());
// thrust_reduce_by_key-NEXT:   /*5*/oneapi::dpl::reduce_by_segment(oneapi::dpl::execution::make_device_policy(q_ct1), d.begin(), d.end(), d.begin(), d.end(), d.begin(), bp);
// thrust_reduce_by_key-NEXT:   /*6*/oneapi::dpl::reduce_by_segment(oneapi::dpl::execution::make_device_policy(q_ct1), d.begin(), d.end(), d.begin(), d.end(), d.begin());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::remove_copy_if --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_remove_copy_if
// thrust_remove_copy_if: CUDA API:
// thrust_remove_copy_if-NEXT:   struct greater_than_zero {
// thrust_remove_copy_if-NEXT:     __host__ __device__ bool operator()(int x) const { return x > 0; }
// thrust_remove_copy_if-NEXT:   };
// thrust_remove_copy_if-NEXT:   thrust::device_vector<int> A(4);
// thrust_remove_copy_if-NEXT:   thrust::device_vector<int> S(4);
// thrust_remove_copy_if-NEXT:   greater_than_zero pred;
// thrust_remove_copy_if-NEXT:   thrust::device_vector<int> d;
// thrust_remove_copy_if-NEXT:   thrust::device_vector<int> R(4);
// thrust_remove_copy_if-NEXT:   /*1*/ thrust::remove_copy_if(thrust::device, A.begin(), A.end(), R.begin(), pred);
// thrust_remove_copy_if-NEXT:   /*2*/ thrust::remove_copy_if(A.begin(), A.end(), R.begin(), pred);
// thrust_remove_copy_if-NEXT:   /*3*/ thrust::remove_copy_if(thrust::device, A.begin(), A.end(), S.begin(),R.begin(), pred);
// thrust_remove_copy_if-NEXT:   /*4*/ thrust::remove_copy_if(A.begin(), A.end(), S.begin(), R.begin(), pred);
// thrust_remove_copy_if-NEXT: Is migrated to:
// thrust_remove_copy_if-NEXT:   struct greater_than_zero {
// thrust_remove_copy_if-NEXT:     bool operator()(int x) const { return x > 0; }
// thrust_remove_copy_if-NEXT:   };
// thrust_remove_copy_if-NEXT:   dpct::device_vector<int> A(4);
// thrust_remove_copy_if-NEXT:   dpct::device_vector<int> S(4);
// thrust_remove_copy_if-NEXT:   greater_than_zero pred;
// thrust_remove_copy_if-NEXT:   dpct::device_vector<int> d;
// thrust_remove_copy_if-NEXT:   dpct::device_vector<int> R(4);
// thrust_remove_copy_if-NEXT:   /*1*/ oneapi::dpl::remove_copy_if(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), R.begin(), pred);
// thrust_remove_copy_if-NEXT:   /*2*/ oneapi::dpl::remove_copy_if(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), R.begin(), pred);
// thrust_remove_copy_if-NEXT:   /*3*/ dpct::remove_copy_if(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), S.begin(), R.begin(), pred);
// thrust_remove_copy_if-NEXT:   /*4*/ dpct::remove_copy_if(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), S.begin(), R.begin(), pred);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::remove_if --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_remove_if
// thrust_remove_if: CUDA API:
// thrust_remove_if-NEXT:   struct greater_than_zero {
// thrust_remove_if-NEXT:     __host__ __device__ bool operator()(int x) const { return x > 0; }
// thrust_remove_if-NEXT:   };
// thrust_remove_if-NEXT:   thrust::device_vector<int> A(4);
// thrust_remove_if-NEXT:   thrust::device_vector<int> S(4);
// thrust_remove_if-NEXT:   greater_than_zero pred;
// thrust_remove_if-NEXT:   thrust::device_vector<int> d;
// thrust_remove_if-NEXT:   /*1*/ thrust::remove_if(thrust::device, A.begin(), A.end(), pred);
// thrust_remove_if-NEXT:   /*2*/ thrust::remove_if(A.begin(), A.end(), pred);
// thrust_remove_if-NEXT:   /*3*/ thrust::remove_if(thrust::device, A.begin(), A.end(), S.begin(), pred);
// thrust_remove_if-NEXT:   /*4*/ thrust::remove_if(A.begin(), A.end(), S.begin(), pred);
// thrust_remove_if-NEXT: Is migrated to:
// thrust_remove_if-NEXT:   struct greater_than_zero {
// thrust_remove_if-NEXT:     bool operator()(int x) const { return x > 0; }
// thrust_remove_if-NEXT:   };
// thrust_remove_if-NEXT:   dpct::device_vector<int> A(4);
// thrust_remove_if-NEXT:   dpct::device_vector<int> S(4);
// thrust_remove_if-NEXT:   greater_than_zero pred;
// thrust_remove_if-NEXT:   dpct::device_vector<int> d;
// thrust_remove_if-NEXT:   /*1*/ oneapi::dpl::remove_if(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), pred);
// thrust_remove_if-NEXT:   /*2*/ oneapi::dpl::remove_if(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), pred);
// thrust_remove_if-NEXT:   /*3*/ dpct::remove_if(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), S.begin(), pred);
// thrust_remove_if-NEXT:   /*4*/ dpct::remove_if(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), S.begin(), pred);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::replace --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_replace
// thrust_replace: CUDA API:
// thrust_replace-NEXT:   thrust::device_vector<int> A(4);
// thrust_replace-NEXT:   thrust::device_vector<int> B(4);
// thrust_replace-NEXT:   /*1*/ thrust::replace(A.begin(), A.end(), 0, 399);
// thrust_replace-NEXT:   /*2*/ thrust::replace(B.begin(), B.end(), 0, 399);
// thrust_replace-NEXT:   /*3*/ thrust::replace(thrust::device, A.begin(), A.end(), 0, 399);
// thrust_replace-NEXT:   /*4*/ thrust::replace(thrust::device, B.begin(), B.end(), 0, 399);
// thrust_replace-NEXT: Is migrated to:
// thrust_replace-NEXT:   dpct::device_vector<int> A(4);
// thrust_replace-NEXT:   dpct::device_vector<int> B(4);
// thrust_replace-NEXT:   /*1*/ oneapi::dpl::replace(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), 0, 399);
// thrust_replace-NEXT:   /*2*/ oneapi::dpl::replace(oneapi::dpl::execution::make_device_policy(q_ct1), B.begin(), B.end(), 0, 399);
// thrust_replace-NEXT:   /*3*/ oneapi::dpl::replace(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), 0, 399);
// thrust_replace-NEXT:   /*4*/ oneapi::dpl::replace(oneapi::dpl::execution::make_device_policy(q_ct1), B.begin(), B.end(), 0, 399);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::replace_copy_if --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_replace_copy_if
// thrust_replace_copy_if: CUDA API:
// thrust_replace_copy_if-NEXT:   struct is_less_than_zero {
// thrust_replace_copy_if-NEXT:     __host__ __device__ bool operator()(int x) const { return x < 0; }
// thrust_replace_copy_if-NEXT:   };
// thrust_replace_copy_if-NEXT:   thrust::host_vector<int> AH(4);
// thrust_replace_copy_if-NEXT:   thrust::host_vector<int> BH(4);
// thrust_replace_copy_if-NEXT:   thrust::host_vector<int> SH(4);
// thrust_replace_copy_if-NEXT:   is_less_than_zero pred;
// thrust_replace_copy_if-NEXT:   /*1*/ thrust::replace_copy_if(AH.begin(), AH.end(), BH.begin(), pred, 0);
// thrust_replace_copy_if-NEXT:   /*2*/ thrust::replace_copy_if(AH.begin(), AH.end(), SH.begin(), BH.begin(), pred,0);
// thrust_replace_copy_if-NEXT:   /*3*/ thrust::replace_copy_if(thrust::host, AH.begin(), AH.end(), BH.begin(), pred,0);
// thrust_replace_copy_if-NEXT:   /*4*/ thrust::replace_copy_if(thrust::host, AH.begin(), AH.end(), SH.begin(),BH.begin(), pred, 0);
// thrust_replace_copy_if-NEXT: Is migrated to:
// thrust_replace_copy_if-NEXT:   struct is_less_than_zero {
// thrust_replace_copy_if-NEXT:     bool operator()(int x) const { return x < 0; }
// thrust_replace_copy_if-NEXT:   };
// thrust_replace_copy_if-NEXT:   std::vector<int> AH(4);
// thrust_replace_copy_if-NEXT:   std::vector<int> BH(4);
// thrust_replace_copy_if-NEXT:   std::vector<int> SH(4);
// thrust_replace_copy_if-NEXT:   is_less_than_zero pred;
// thrust_replace_copy_if-NEXT:   /*1*/ oneapi::dpl::replace_copy_if(oneapi::dpl::execution::seq, AH.begin(), AH.end(), BH.begin(), pred, 0);
// thrust_replace_copy_if-NEXT:   /*2*/ dpct::replace_copy_if(oneapi::dpl::execution::seq, AH.begin(), AH.end(), SH.begin(), BH.begin(), pred, 0);
// thrust_replace_copy_if-NEXT:   /*3*/ oneapi::dpl::replace_copy_if(oneapi::dpl::execution::seq, AH.begin(), AH.end(), BH.begin(), pred, 0);
// thrust_replace_copy_if-NEXT:   /*4*/ dpct::replace_copy_if(oneapi::dpl::execution::seq, AH.begin(), AH.end(), SH.begin(), BH.begin(), pred, 0);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::replace_if --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_replace_if
// thrust_replace_if: CUDA API:
// thrust_replace_if-NEXT:   struct is_less_than_zero {
// thrust_replace_if-NEXT:     __host__ __device__ bool operator()(int x) const { return x < 0; }
// thrust_replace_if-NEXT:   };
// thrust_replace_if-NEXT:   thrust::host_vector<int> AH(4);
// thrust_replace_if-NEXT:   thrust::host_vector<int> BH(4);
// thrust_replace_if-NEXT:   thrust::host_vector<int> SH(4);
// thrust_replace_if-NEXT:   is_less_than_zero pred;
// thrust_replace_if-NEXT:   /*1*/ thrust::replace_if(AH.begin(), AH.end(), pred, 0);
// thrust_replace_if-NEXT:   /*2*/ thrust::replace_if(AH.begin(), AH.end(), SH.begin(), pred, 0);
// thrust_replace_if-NEXT:   /*3*/ thrust::replace_if(thrust::host, AH.begin(), AH.end(), pred, 0);
// thrust_replace_if-NEXT:   /*4*/ thrust::replace_if(thrust::host, AH.begin(), AH.end(), SH.begin(), pred, 0);
// thrust_replace_if-NEXT: Is migrated to:
// thrust_replace_if-NEXT:   struct is_less_than_zero {
// thrust_replace_if-NEXT:     bool operator()(int x) const { return x < 0; }
// thrust_replace_if-NEXT:   };
// thrust_replace_if-NEXT:   std::vector<int> AH(4);
// thrust_replace_if-NEXT:   std::vector<int> BH(4);
// thrust_replace_if-NEXT:   std::vector<int> SH(4);
// thrust_replace_if-NEXT:   is_less_than_zero pred;
// thrust_replace_if-NEXT:   /*1*/ oneapi::dpl::replace_if(oneapi::dpl::execution::seq, AH.begin(), AH.end(), pred, 0);
// thrust_replace_if-NEXT:   /*2*/ dpct::replace_if(oneapi::dpl::execution::seq, AH.begin(), AH.end(), SH.begin(), pred, 0);
// thrust_replace_if-NEXT:   /*3*/ oneapi::dpl::replace_if(oneapi::dpl::execution::seq, AH.begin(), AH.end(), pred, 0);
// thrust_replace_if-NEXT:   /*4*/ dpct::replace_if(oneapi::dpl::execution::seq, AH.begin(), AH.end(), SH.begin(), pred, 0);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::return_temporary_buffer --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_return_temporary_buffer
// thrust_return_temporary_buffer: CUDA API:
// thrust_return_temporary_buffer-NEXT:   const int N = 100;
// thrust_return_temporary_buffer-NEXT:   typedef thrust::pair<thrust::pointer<int, thrust::device_system_tag>, std::ptrdiff_t> ptr_and_size_t;
// thrust_return_temporary_buffer-NEXT:   thrust::device_system_tag device_sys;
// thrust_return_temporary_buffer-NEXT:   ptr_and_size_t ptr_and_size = thrust::get_temporary_buffer<int>(device_sys, N);
// thrust_return_temporary_buffer-NEXT:   thrust::return_temporary_buffer(device_sys, ptr_and_size.first,ptr_and_size.second);
// thrust_return_temporary_buffer-NEXT: Is migrated to:
// thrust_return_temporary_buffer-NEXT:   const int N = 100;
// thrust_return_temporary_buffer-NEXT:   typedef std::pair<dpct::tagged_pointer<int, dpct::device_sys_tag>, std::ptrdiff_t> ptr_and_size_t;
// thrust_return_temporary_buffer-NEXT:   dpct::device_sys_tag device_sys;
// thrust_return_temporary_buffer-NEXT:   ptr_and_size_t ptr_and_size = dpct::get_temporary_allocation<int>(device_sys, N);
// thrust_return_temporary_buffer-NEXT:   dpct::release_temporary_allocation(device_sys, ptr_and_size.first);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::scatter --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_scatter
// thrust_scatter: CUDA API:
// thrust_scatter-NEXT:   int values[10] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
// thrust_scatter-NEXT:   thrust::host_vector<int> h_values(values, values + 10);
// thrust_scatter-NEXT:   int map[10] = {0, 5, 1, 6, 2, 7, 3, 8, 4, 9};
// thrust_scatter-NEXT:   thrust::host_vector<int> h_map(map, map + 10);
// thrust_scatter-NEXT:   thrust::host_vector<int> h_output(10);
// thrust_scatter-NEXT:   /*1*/ thrust::scatter(thrust::seq, h_values.begin(), h_values.end(),h_map.begin(), h_output.begin());
// thrust_scatter-NEXT:   /*2*/ thrust::scatter(h_values.begin(), h_values.end(), h_map.begin(),h_output.begin());
// thrust_scatter-NEXT: Is migrated to:
// thrust_scatter-NEXT:   int values[10] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
// thrust_scatter-NEXT:   std::vector<int> h_values(values, values + 10);
// thrust_scatter-NEXT:   int map[10] = {0, 5, 1, 6, 2, 7, 3, 8, 4, 9};
// thrust_scatter-NEXT:   std::vector<int> h_map(map, map + 10);
// thrust_scatter-NEXT:   std::vector<int> h_output(10);
// thrust_scatter-NEXT:   /*1*/ dpct::scatter(oneapi::dpl::execution::seq, h_values.begin(), h_values.end(), h_map.begin(), h_output.begin());
// thrust_scatter-NEXT:   /*2*/ dpct::scatter(oneapi::dpl::execution::seq, h_values.begin(), h_values.end(), h_map.begin(), h_output.begin());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::sequence --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_sequence
// thrust_sequence: CUDA API:
// thrust_sequence-NEXT:   const int N = 5;
// thrust_sequence-NEXT:   thrust::device_vector<int> vec(N);
// thrust_sequence-NEXT:   thrust::sequence(vec.begin(), vec.end());
// thrust_sequence-NEXT: Is migrated to:
// thrust_sequence-NEXT:   const int N = 5;
// thrust_sequence-NEXT:   dpct::device_vector<int> vec(N);
// thrust_sequence-NEXT:   dpct::iota(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), vec.begin(), vec.end());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::sort_by_key --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_sort_by_key
// thrust_sort_by_key: CUDA API:
// thrust_sort_by_key-NEXT:   thrust::host_vector<int> h;
// thrust_sort_by_key-NEXT:   thrust::device_vector<int> d;
// thrust_sort_by_key-NEXT:   /*1*/ thrust::sort_by_key(thrust::seq, h.begin(), h.end(), h.begin(), thrust::greater<int>());
// thrust_sort_by_key-NEXT:   /*2*/ thrust::sort_by_key(h.begin(), h.end(), h.begin());
// thrust_sort_by_key-NEXT:   /*3*/ thrust::sort_by_key(thrust::device, d.begin(), d.end(), d.begin());
// thrust_sort_by_key-NEXT:   /*4*/ thrust::sort_by_key(h.begin(), h.end(), h.begin(), thrust::greater<int>());
// thrust_sort_by_key-NEXT: Is migrated to:
// thrust_sort_by_key-NEXT:   std::vector<int> h;
// thrust_sort_by_key-NEXT:   dpct::device_vector<int> d;
// thrust_sort_by_key-NEXT:   /*1*/ dpct::sort(oneapi::dpl::execution::seq, h.begin(), h.end(), h.begin(), std::greater<int>());
// thrust_sort_by_key-NEXT:   /*2*/ dpct::sort(oneapi::dpl::execution::seq, h.begin(), h.end(), h.begin());
// thrust_sort_by_key-NEXT:   /*3*/ dpct::sort(oneapi::dpl::execution::make_device_policy(q_ct1), d.begin(), d.end(), d.begin());
// thrust_sort_by_key-NEXT:   /*4*/ dpct::sort(oneapi::dpl::execution::seq, h.begin(), h.end(), h.begin(), std::greater<int>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::stable_sort_by_key --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_stable_sort_by_key
// thrust_stable_sort_by_key: CUDA API:
// thrust_stable_sort_by_key-NEXT:   thrust::device_vector<int> AD(4);
// thrust_stable_sort_by_key-NEXT:   thrust::device_vector<int> BD(4);
// thrust_stable_sort_by_key-NEXT:   thrust::host_vector<int> AH(4);
// thrust_stable_sort_by_key-NEXT:   thrust::host_vector<int> BH(4);
// thrust_stable_sort_by_key-NEXT:   /*1*/ thrust::stable_sort_by_key(                AH.begin(), AH.end(), BH.begin());
// thrust_stable_sort_by_key-NEXT:   /*2*/ thrust::stable_sort_by_key(                AH.begin(), AH.end(), BH.begin(), thrust::greater<int>());
// thrust_stable_sort_by_key-NEXT:   /*3*/ thrust::stable_sort_by_key(thrust::host,   AH.begin(), AH.end(), BH.begin());
// thrust_stable_sort_by_key-NEXT:   /*4*/ thrust::stable_sort_by_key(thrust::host,   AH.begin(), AH.end(), BH.begin(), thrust::greater<int>());
// thrust_stable_sort_by_key-NEXT: Is migrated to:
// thrust_stable_sort_by_key-NEXT:   dpct::device_vector<int> AD(4);
// thrust_stable_sort_by_key-NEXT:   dpct::device_vector<int> BD(4);
// thrust_stable_sort_by_key-NEXT:   std::vector<int> AH(4);
// thrust_stable_sort_by_key-NEXT:   std::vector<int> BH(4);
// thrust_stable_sort_by_key-NEXT:   /*1*/ dpct::stable_sort(oneapi::dpl::execution::seq, AH.begin(), AH.end(), BH.begin());
// thrust_stable_sort_by_key-NEXT:   /*2*/ dpct::stable_sort(oneapi::dpl::execution::seq, AH.begin(), AH.end(), BH.begin(), std::greater<int>());
// thrust_stable_sort_by_key-NEXT:   /*3*/ dpct::stable_sort(oneapi::dpl::execution::seq, AH.begin(), AH.end(), BH.begin());
// thrust_stable_sort_by_key-NEXT:   /*4*/ dpct::stable_sort(oneapi::dpl::execution::seq, AH.begin(), AH.end(), BH.begin(), std::greater<int>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::transform --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_transform
// thrust_transform: CUDA API:
// thrust_transform-NEXT:   const int N = 1000;
// thrust_transform-NEXT:   thrust::device_vector<float> t1(N);
// thrust_transform-NEXT:   thrust::device_vector<float> t2(N);
// thrust_transform-NEXT:   thrust::device_vector<float> t3(N);
// thrust_transform-NEXT:   thrust::transform(t1.begin(), t1.end(), t2.begin(), t3.begin(), thrust::divides<float>());
// thrust_transform-NEXT: Is migrated to:
// thrust_transform-NEXT:   const int N = 1000;
// thrust_transform-NEXT:   dpct::device_vector<float> t1(N);
// thrust_transform-NEXT:   dpct::device_vector<float> t2(N);
// thrust_transform-NEXT:   dpct::device_vector<float> t3(N);
// thrust_transform-NEXT:   std::transform(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), t1.begin(), t1.end(), t2.begin(), t3.begin(), std::divides<float>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::uninitialized_fill --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_uninitialized_fill
// thrust_uninitialized_fill: CUDA API:
// thrust_uninitialized_fill-NEXT:   thrust::host_vector<int> h_input(10);
// thrust_uninitialized_fill-NEXT:   thrust::uninitialized_fill(h_input.begin(), h_input.end(), 10);
// thrust_uninitialized_fill-NEXT: Is migrated to:
// thrust_uninitialized_fill-NEXT:   std::vector<int> h_input(10);
// thrust_uninitialized_fill-NEXT:   std::uninitialized_fill(oneapi::dpl::execution::seq, h_input.begin(), h_input.end(), 10);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::unique --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_unique
// thrust_unique: CUDA API:
// thrust_unique-NEXT:   thrust::host_vector<int> h_input(10);
// thrust_unique-NEXT:   thrust::unique(h_input.begin(), h_input.end());
// thrust_unique-NEXT: Is migrated to:
// thrust_unique-NEXT:   std::vector<int> h_input(10);
// thrust_unique-NEXT:   std::unique(oneapi::dpl::execution::seq, h_input.begin(), h_input.end());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::unique_by_key_copy --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_unique_by_key_copy
// thrust_unique_by_key_copy: CUDA API:
// thrust_unique_by_key_copy-NEXT:   const int N = 7;
// thrust_unique_by_key_copy-NEXT:   int A[N] = {1, 3, 3, 3, 2, 2, 1}; // input keys
// thrust_unique_by_key_copy-NEXT:   int B[N] = {9, 8, 7, 6, 5, 4, 3}; // input values
// thrust_unique_by_key_copy-NEXT:   thrust::device_vector<int> d_keys(A, A + N);
// thrust_unique_by_key_copy-NEXT:   thrust::device_vector<int> d_values(B, B + N);
// thrust_unique_by_key_copy-NEXT:   thrust::device_vector<int> d_output_keys(N);
// thrust_unique_by_key_copy-NEXT:   thrust::device_vector<int> d_output_values(N);
// thrust_unique_by_key_copy-NEXT:   thrust::equal_to<int> binary_pred;
// thrust_unique_by_key_copy-NEXT:   typedef thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> iter_pair;
// thrust_unique_by_key_copy-NEXT:   thrust::device_vector<iter_pair> new_last_vec(1);
// thrust_unique_by_key_copy-NEXT:   /*1*/ *new_last_vec.begin() = thrust::unique_by_key_copy(thrust::device, d_keys.begin(), d_keys.end(), d_values.begin(), d_output_keys.begin(), d_output_values.begin(), binary_pred);
// thrust_unique_by_key_copy-NEXT:   /*2*/ *new_last_vec.begin() = thrust::unique_by_key_copy(d_keys.begin(), d_keys.end(), d_values.begin(), d_output_keys.begin(), d_output_values.begin(), binary_pred);
// thrust_unique_by_key_copy-NEXT:   /*3*/ *new_last_vec.begin() = thrust::unique_by_key_copy(d_keys.begin(), d_keys.end(), d_values.begin(), d_output_keys.begin(), d_output_values.begin());
// thrust_unique_by_key_copy-NEXT:   /*4*/ *new_last_vec.begin() = thrust::unique_by_key_copy(thrust::device, d_keys.begin(), d_keys.end(), d_values.begin(), d_output_keys.begin(), d_output_values.begin());
// thrust_unique_by_key_copy-NEXT: Is migrated to:
// thrust_unique_by_key_copy-NEXT:   const int N = 7;
// thrust_unique_by_key_copy-NEXT:   int A[N] = {1, 3, 3, 3, 2, 2, 1}; // input keys
// thrust_unique_by_key_copy-NEXT:   int B[N] = {9, 8, 7, 6, 5, 4, 3}; // input values
// thrust_unique_by_key_copy-NEXT:   dpct::device_vector<int> d_keys(A, A + N);
// thrust_unique_by_key_copy-NEXT:   dpct::device_vector<int> d_values(B, B + N);
// thrust_unique_by_key_copy-NEXT:   dpct::device_vector<int> d_output_keys(N);
// thrust_unique_by_key_copy-NEXT:   dpct::device_vector<int> d_output_values(N);
// thrust_unique_by_key_copy-NEXT:   oneapi::dpl::equal_to<int> binary_pred;
// thrust_unique_by_key_copy-NEXT:   typedef std::pair<dpct::device_vector<int>::iterator, dpct::device_vector<int>::iterator> iter_pair;
// thrust_unique_by_key_copy-NEXT:   dpct::device_vector<iter_pair> new_last_vec(1);
// thrust_unique_by_key_copy-NEXT:   /*1*/ *new_last_vec.begin() = dpct::unique_copy(oneapi::dpl::execution::make_device_policy(q_ct1), d_keys.begin(), d_keys.end(), d_values.begin(), d_output_keys.begin(), d_output_values.begin(), binary_pred);
// thrust_unique_by_key_copy-NEXT:   /*2*/ *new_last_vec.begin() = dpct::unique_copy(oneapi::dpl::execution::make_device_policy(q_ct1), d_keys.begin(), d_keys.end(), d_values.begin(), d_output_keys.begin(), d_output_values.begin(), binary_pred);
// thrust_unique_by_key_copy-NEXT:   /*3*/ *new_last_vec.begin() = dpct::unique_copy(oneapi::dpl::execution::make_device_policy(q_ct1), d_keys.begin(), d_keys.end(), d_values.begin(), d_output_keys.begin(), d_output_values.begin());
// thrust_unique_by_key_copy-NEXT:   /*4*/ *new_last_vec.begin() = dpct::unique_copy(oneapi::dpl::execution::make_device_policy(q_ct1), d_keys.begin(), d_keys.end(), d_values.begin(), d_output_keys.begin(), d_output_values.begin());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::upper_bound --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_upper_bound
// thrust_upper_bound: CUDA API:
// thrust_upper_bound-NEXT:   std::vector<int> v, v2, v3, v4;
// thrust_upper_bound-NEXT:   auto bp = [](int x, int y) -> bool { return x < y; };
// thrust_upper_bound-NEXT:   /*1*/ thrust::upper_bound(thrust::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
// thrust_upper_bound-NEXT:   /*2*/ thrust::upper_bound(v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
// thrust_upper_bound-NEXT:   /*3*/ thrust::upper_bound(thrust::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
// thrust_upper_bound-NEXT:   /*4*/ thrust::upper_bound(v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
// thrust_upper_bound-NEXT: Is migrated to:
// thrust_upper_bound-NEXT:   std::vector<int> v, v2, v3, v4;
// thrust_upper_bound-NEXT:   auto bp = [](int x, int y) -> bool { return x < y; };
// thrust_upper_bound-NEXT:   /*1*/ oneapi::dpl::upper_bound(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
// thrust_upper_bound-NEXT:   /*2*/ oneapi::dpl::upper_bound(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin());
// thrust_upper_bound-NEXT:   /*3*/ oneapi::dpl::upper_bound(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);
// thrust_upper_bound-NEXT:   /*4*/ oneapi::dpl::upper_bound(oneapi::dpl::execution::seq, v.begin(), v.end(), v2.begin(), v2.end(), v3.begin(), bp);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::transform_reduce --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_transform_reduce
// thrust_transform_reduce: CUDA API:
// thrust_transform_reduce-NEXT:   int data[10];
// thrust_transform_reduce-NEXT:   cudaStream_t stream;
// thrust_transform_reduce-NEXT:   thrust::device_ptr<int> begin = thrust::device_pointer_cast(&data[0]);
// thrust_transform_reduce-NEXT:   thrust::device_ptr<int> end = begin + 10;
// thrust_transform_reduce-NEXT:   /*1*/ bool h_result = thrust::transform_reduce(begin, end, square, 0, thrust::plus<bool>());
// thrust_transform_reduce-NEXT:   /*2*/ bool h_result_1 = thrust::transform_reduce(thrust::seq, begin, end, square, 0, thrust::plus<bool>());
// thrust_transform_reduce-NEXT:   /*3*/ bool h_result_2 = thrust::transform_reduce(thrust::cuda::par.on(stream), begin, end, square, 0, thrust::plus<bool>());
// thrust_transform_reduce-NEXT: Is migrated to:
// thrust_transform_reduce-NEXT:   int data[10];
// thrust_transform_reduce-NEXT:   dpct::queue_ptr stream;
// thrust_transform_reduce-NEXT:   dpct::device_pointer<int> begin = dpct::get_device_pointer(&data[0]);
// thrust_transform_reduce-NEXT:   dpct::device_pointer<int> end = begin + 10;
// thrust_transform_reduce-NEXT:   /*1*/ bool h_result = std::transform_reduce(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), begin, end, 0, std::plus<bool>(), square);
// thrust_transform_reduce-NEXT:   /*2*/ bool h_result_1 = std::transform_reduce(oneapi::dpl::execution::seq, begin, end, 0, std::plus<bool>(), square);
// thrust_transform_reduce-NEXT:   /*3*/ bool h_result_2 = std::transform_reduce(oneapi::dpl::execution::make_device_policy(*stream), begin, end, 0, std::plus<bool>(), square);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::copy --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_copy
// thrust_copy: CUDA API:
// thrust_copy-NEXT:   const char data[] = "abc";
// thrust_copy-NEXT:   const size_t N = (sizeof(data) / sizeof(char)) - 1;
// thrust_copy-NEXT:   char dst_data[N];
// thrust_copy-NEXT:   thrust::device_vector<char> input(data, data + N);
// thrust_copy-NEXT:   thrust::device_vector<char> dst(data, data + N);
// thrust_copy-NEXT:   /*1*/ thrust::copy(data, data + N, dst_data);
// thrust_copy-NEXT:   /*2*/ thrust::copy(thrust::device, input.begin(), input.begin() + N, dst.begin());
// thrust_copy-NEXT: Is migrated to:
// thrust_copy-NEXT:   const char data[] = "abc";
// thrust_copy-NEXT:   const size_t N = (sizeof(data) / sizeof(char)) - 1;
// thrust_copy-NEXT:   char dst_data[N];
// thrust_copy-NEXT:   dpct::device_vector<char> input(data, data + N);
// thrust_copy-NEXT:   dpct::device_vector<char> dst(data, data + N);
// thrust_copy-NEXT:   /*1*/ std::copy(oneapi::dpl::execution::seq, data, data + N, dst_data);
// thrust_copy-NEXT:   /*2*/ std::copy(oneapi::dpl::execution::make_device_policy(q_ct1), input.begin(), input.begin() + N, dst.begin());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::copy_n --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_copy_n
// thrust_copy_n: CUDA API:
// thrust_copy_n-NEXT:   const char data[] = "abc";
// thrust_copy_n-NEXT:   const size_t N = (sizeof(data) / sizeof(char)) - 1;
// thrust_copy_n-NEXT:   char dst_data[N];
// thrust_copy_n-NEXT:   thrust::device_vector<char> input(data, data + N);
// thrust_copy_n-NEXT:   thrust::device_vector<char> dst(data, data + N);
// thrust_copy_n-NEXT:   /*1*/ thrust::copy_n(data, N, dst_data);
// thrust_copy_n-NEXT:   /*2*/ thrust::copy_n(thrust::device, input.begin(), N, dst.begin());
// thrust_copy_n-NEXT: Is migrated to:
// thrust_copy_n-NEXT:   const char data[] = "abc";
// thrust_copy_n-NEXT:   const size_t N = (sizeof(data) / sizeof(char)) - 1;
// thrust_copy_n-NEXT:   char dst_data[N];
// thrust_copy_n-NEXT:   dpct::device_vector<char> input(data, data + N);
// thrust_copy_n-NEXT:   dpct::device_vector<char> dst(data, data + N);
// thrust_copy_n-NEXT:   /*1*/ std::copy_n(oneapi::dpl::execution::seq, data, N, dst_data);
// thrust_copy_n-NEXT:   /*2*/ std::copy_n(oneapi::dpl::execution::make_device_policy(q_ct1), input.begin(), N, dst.begin());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::gather_if --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_gather_if
// thrust_gather_if: CUDA API:
// thrust_gather_if-NEXT:   thrust::host_vector<int> AH(4);
// thrust_gather_if-NEXT:   thrust::host_vector<int> BH(4);
// thrust_gather_if-NEXT:   thrust::host_vector<int> SH(4);
// thrust_gather_if-NEXT:   thrust::host_vector<int> RH(4);
// thrust_gather_if-NEXT:   struct is_less_than_zero {
// thrust_gather_if-NEXT:     __host__ __device__ bool operator()(int x) const { return x < 0; }
// thrust_gather_if-NEXT:   };
// thrust_gather_if-NEXT:   is_less_than_zero pred;
// thrust_gather_if-NEXT:   /*1*/ thrust::gather_if(AH.begin(), AH.end(), SH.begin(), BH.begin(), RH.begin(),pred);
// thrust_gather_if-NEXT:   /*2*/ thrust::gather_if(thrust::host, AH.begin(), AH.end(), SH.begin(), BH.begin(),RH.begin(), pred);
// thrust_gather_if-NEXT:   /*3*/ thrust::gather_if(AH.begin(), AH.end(), SH.begin(), BH.begin(), RH.begin());
// thrust_gather_if-NEXT:   /*4*/ thrust::gather_if(thrust::host, AH.begin(), AH.end(), SH.begin(), BH.begin(),RH.begin());
// thrust_gather_if-NEXT: Is migrated to:
// thrust_gather_if-NEXT:   std::vector<int> AH(4);
// thrust_gather_if-NEXT:   std::vector<int> BH(4);
// thrust_gather_if-NEXT:   std::vector<int> SH(4);
// thrust_gather_if-NEXT:   std::vector<int> RH(4);
// thrust_gather_if-NEXT:   struct is_less_than_zero {
// thrust_gather_if-NEXT:     bool operator()(int x) const { return x < 0; }
// thrust_gather_if-NEXT:   };
// thrust_gather_if-NEXT:   is_less_than_zero pred;
// thrust_gather_if-NEXT:   /*1*/ dpct::gather_if(oneapi::dpl::execution::seq, AH.begin(), AH.end(), SH.begin(), BH.begin(), RH.begin(), pred);
// thrust_gather_if-NEXT:   /*2*/ dpct::gather_if(oneapi::dpl::execution::seq, AH.begin(), AH.end(), SH.begin(), BH.begin(), RH.begin(), pred);
// thrust_gather_if-NEXT:   /*3*/ dpct::gather_if(oneapi::dpl::execution::seq, AH.begin(), AH.end(), SH.begin(), BH.begin(), RH.begin());
// thrust_gather_if-NEXT:   /*4*/ dpct::gather_if(oneapi::dpl::execution::seq, AH.begin(), AH.end(), SH.begin(), BH.begin(), RH.begin());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::transform_if --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_transform_if
// thrust_transform_if: CUDA API:
// thrust_transform_if-NEXT:   struct is_odd {
// thrust_transform_if-NEXT:     __host__ __device__ bool operator()(const int x) const {
// thrust_transform_if-NEXT:       return x % 2;
// thrust_transform_if-NEXT:     }
// thrust_transform_if-NEXT:   };
// thrust_transform_if-NEXT:   struct identity {
// thrust_transform_if-NEXT:     __host__ __device__ bool operator()(const int x) const {
// thrust_transform_if-NEXT:       return x;
// thrust_transform_if-NEXT:     }
// thrust_transform_if-NEXT:   };
// thrust_transform_if-NEXT:   thrust::negate<int> neg;
// thrust_transform_if-NEXT:   thrust::plus<int> plus;
// thrust_transform_if-NEXT:   const int dataLen = 10;
// thrust_transform_if-NEXT:   int inDataH[dataLen]  = {-5, 0, 2, -3, 2, 4, 0, -1, 2, 8};
// thrust_transform_if-NEXT:   int outDataH[dataLen];
// thrust_transform_if-NEXT:   int stencilH[dataLen] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
// thrust_transform_if-NEXT:   thrust::device_vector<int> inDataD(dataLen);
// thrust_transform_if-NEXT:   thrust::device_vector<int> outDataD(dataLen);
// thrust_transform_if-NEXT:   thrust::device_vector<int> stencilD(dataLen);
// thrust_transform_if-NEXT:   /*1*/ thrust::transform_if(thrust::host, inDataH, inDataH + dataLen, outDataH, neg, is_odd());
// thrust_transform_if-NEXT:   /*2*/ thrust::transform_if(thrust::host, inDataH, inDataH + dataLen, stencilH, outDataH, neg, identity());
// thrust_transform_if-NEXT:   /*3*/ thrust::transform_if(thrust::host, inDataH, inDataH + dataLen, inDataH, stencilH, outDataH, plus, identity());
// thrust_transform_if-NEXT:   /*4*/ thrust::transform_if(inDataD.begin(), inDataD.end(), outDataD.begin(), neg, is_odd());
// thrust_transform_if-NEXT:   /*5*/ thrust::transform_if(inDataD.begin(), inDataD.end(), stencilD.begin(), outDataD.begin(), neg, identity());
// thrust_transform_if-NEXT:   /*6*/ thrust::transform_if(inDataD.begin(), inDataD.end(), inDataD.begin(), stencilD.begin(), outDataD.begin(), plus, identity());
// thrust_transform_if-NEXT: Is migrated to:
// thrust_transform_if-NEXT:   struct is_odd {
// thrust_transform_if-NEXT:     bool operator()(const int x) const {
// thrust_transform_if-NEXT:       return x % 2;
// thrust_transform_if-NEXT:     }
// thrust_transform_if-NEXT:   };
// thrust_transform_if-NEXT:   struct identity {
// thrust_transform_if-NEXT:     bool operator()(const int x) const {
// thrust_transform_if-NEXT:       return x;
// thrust_transform_if-NEXT:     }
// thrust_transform_if-NEXT:   };
// thrust_transform_if-NEXT:   std::negate<int> neg;
// thrust_transform_if-NEXT:   std::plus<int> plus;
// thrust_transform_if-NEXT:   const int dataLen = 10;
// thrust_transform_if-NEXT:   int inDataH[dataLen]  = {-5, 0, 2, -3, 2, 4, 0, -1, 2, 8};
// thrust_transform_if-NEXT:   int outDataH[dataLen];
// thrust_transform_if-NEXT:   int stencilH[dataLen] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
// thrust_transform_if-NEXT:   dpct::device_vector<int> inDataD(dataLen);
// thrust_transform_if-NEXT:   dpct::device_vector<int> outDataD(dataLen);
// thrust_transform_if-NEXT:   dpct::device_vector<int> stencilD(dataLen);
// thrust_transform_if-NEXT:   /*1*/ dpct::transform_if(oneapi::dpl::execution::seq, inDataH, inDataH + dataLen, outDataH, neg, is_odd());
// thrust_transform_if-NEXT:   /*2*/ dpct::transform_if(oneapi::dpl::execution::seq, inDataH, inDataH + dataLen, stencilH, outDataH, neg, identity());
// thrust_transform_if-NEXT:   /*3*/ dpct::transform_if(oneapi::dpl::execution::seq, inDataH, inDataH + dataLen, inDataH, stencilH, outDataH, plus, identity());
// thrust_transform_if-NEXT:   /*4*/ dpct::transform_if(oneapi::dpl::execution::make_device_policy(q_ct1), inDataD.begin(), inDataD.end(), outDataD.begin(), neg, is_odd());
// thrust_transform_if-NEXT:   /*5*/ dpct::transform_if(oneapi::dpl::execution::make_device_policy(q_ct1), inDataD.begin(), inDataD.end(), stencilD.begin(), outDataD.begin(), neg, identity());
// thrust_transform_if-NEXT:   /*6*/ dpct::transform_if(oneapi::dpl::execution::make_device_policy(q_ct1), inDataD.begin(), inDataD.end(), inDataD.begin(), stencilD.begin(), outDataD.begin(), plus, identity());
