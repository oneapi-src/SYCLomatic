// UNSUPPORTED: system-windows
// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1
// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::abs --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_abs
// thrust_abs:CUDA API:
// thrust_abs-NEXT:  thrust::abs(thrust::complex<float>(0.0));
// thrust_abs-NEXT:Is migrated to:
// thrust_abs-NEXT:  std::abs(std::complex<float>(0.0));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::log10 --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_log10
// thrust_log10:CUDA API:
// thrust_log10-NEXT:   thrust::log10(thrust::complex<float>(0.0));
// thrust_log10-NEXT:Is migrated to:
// thrust_log10-NEXT:   std::log10(std::complex<float>(0.0));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::sqrt --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_sqrt
// thrust_sqrt:CUDA API:
// thrust_sqrt-NEXT:  thrust::sqrt(thrust::complex<float>(0.0));
// thrust_sqrt-NEXT:Is migrated to:
// thrust_sqrt-NEXT:  std::sqrt(std::complex<float>(0.0));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::sinh --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_sinh
// thrust_sinh:CUDA API:
// thrust_sinh-NEXT:  thrust::sinh(thrust::complex<float>(0.0));
// thrust_sinh-NEXT:Is migrated to:
// thrust_sinh-NEXT:  std::sinh(std::complex<float>(0.0));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::cosh --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_cosh
// thrust_cosh:CUDA API:
// thrust_cosh-NEXT:    thrust::cosh(thrust::complex<float>(0.0));
// thrust_cosh-NEXT:Is migrated to:
// thrust_cosh-NEXT:    std::cosh(std::complex<float>(0.0));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::tanh --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_tanh
// thrust_tanh:CUDA API:
// thrust_tanh-NEXT:  thrust::tanh(thrust::complex<float>(0.0));
// thrust_tanh-NEXT:Is migrated to:
// thrust_tanh-NEXT:  std::tanh(std::complex<float>(0.0));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::asinh --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_asinh
// thrust_asinh:CUDA API:
// thrust_asinh-NEXT:  thrust::asinh(thrust::complex<float>(0.0));
// thrust_asinh-NEXT:Is migrated to:
// thrust_asinh-NEXT:  std::asinh(std::complex<float>(0.0));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::acosh --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_acosh
// thrust_acosh:CUDA API:
// thrust_acosh-NEXT:  thrust::acosh(thrust::complex<float>(0.0));
// thrust_acosh-NEXT:Is migrated to:
// thrust_acosh-NEXT:  std::acosh(std::complex<float>(0.0));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::atanh --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_atanh
// thrust_atanh:CUDA API:
// thrust_atanh-NEXT:  thrust::atanh(thrust::complex<float>(0.0));
// thrust_atanh-NEXT:Is migrated to:
// thrust_atanh-NEXT:  std::atanh(std::complex<float>(0.0));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::polar --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_polar
// thrust_polar:CUDA API:
// thrust_polar-NEXT:  thrust::polar(1.0, 1.0);
// thrust_polar-NEXT:Is migrated to:
// thrust_polar-NEXT:  std::polar(1.0, 1.0);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::exp --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_exp
// thrust_exp:CUDA API:
// thrust_exp-NEXT:  thrust::exp(thrust::complex<float>(0.0));
// thrust_exp-NEXT:Is migrated to:
// thrust_exp-NEXT:  std::exp(std::complex<float>(0.0));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::log --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_log
// thrust_log:CUDA API:
// thrust_log-NEXT:  thrust::log(thrust::complex<float>(0.0));
// thrust_log-NEXT:Is migrated to:
// thrust_log-NEXT:  std::log(std::complex<float>(0.0));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::norm --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_norm
// thrust_norm:CUDA API:
// thrust_norm-NEXT:  thrust::norm(thrust::complex<float>(0.0));
// thrust_norm-NEXT:Is migrated to:
// thrust_norm-NEXT:  std::norm(std::complex<float>(0.0));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::conj --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_conj
// thrust_conj:CUDA API:
// thrust_conj-NEXT:  thrust::conj(thrust::complex<float>(0.0));
// thrust_conj-NEXT:Is migrated to:
// thrust_conj-NEXT:  std::conj(std::complex<float>(0.0));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::proj --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_proj
// thrust_proj:CUDA API:
// thrust_proj-NEXT:    thrust::proj(1);
// thrust_proj-NEXT:Is migrated to:
// thrust_proj-NEXT:    std::proj(1);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::atan --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_atan
// thrust_atan:CUDA API:
// thrust_atan-NEXT:  thrust::atan(thrust::complex<float>(0.0));
// thrust_atan-NEXT:Is migrated to:
// thrust_atan-NEXT:  std::atan(std::complex<float>(0.0));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::advance --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_advance
// thrust_advance:CUDA API:
// thrust_advance-NEXT:  thrust::device_vector<int> vec(N);
// thrust_advance-NEXT:  thrust::device_vector<int>::iterator iter = vec.begin();
// thrust_advance-NEXT:  thrust::advance(iter, 2);
// thrust_advance-NEXT:Is migrated to:
// thrust_advance-NEXT:  dpct::device_vector<int> vec(N);
// thrust_advance-NEXT:  dpct::device_vector<int>::iterator iter = vec.begin();
// thrust_advance-NEXT:  std::advance(iter, 2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::arg --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_arg
// thrust_arg:CUDA API:
// thrust_arg-NEXT:   thrust::arg(thrust::complex<double>(1.0));
// thrust_arg-NEXT:Is migrated to:
// thrust_arg-NEXT:   std::arg(std::complex<double>(1.0));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::distance --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_distance
// thrust_distance: CUDA API:
// thrust_distance-NEXT:  thrust::device_vector<int> vec(13);
// thrust_distance-NEXT:  thrust::device_vector<int>::iterator iter1 = vec.begin();
// thrust_distance-NEXT:  thrust::device_vector<int>::iterator iter2 = iter1 + 7;
// thrust_distance-NEXT:  int d = thrust::distance(iter1, iter2);
// thrust_distance-NEXT:Is migrated to:
// thrust_distance-NEXT:  dpct::device_vector<int> vec(13);
// thrust_distance-NEXT:  dpct::device_vector<int>::iterator iter1 = vec.begin();
// thrust_distance-NEXT:  dpct::device_vector<int>::iterator iter2 = iter1 + 7;
// thrust_distance-NEXT:  int d = oneapi::dpl::distance(iter1, iter2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::tie --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_tie
// thrust_tie:CUDA API:
// thrust_tie-NEXT:  double a, b;
// thrust_tie-NEXT:  thrust::tie(a, b) = thrust::make_tuple(1.0, 2.0);
// thrust_tie-NEXT:Is migrated to:
// thrust_tie-NEXT:  double a, b;
// thrust_tie-NEXT:  std::tie(a, b) = std::make_tuple(1.0, 2.0);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::not1 --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_not1
// thrust_not1:CUDA API:
// thrust_not1-NEXT:  struct greater_than_zero {
// thrust_not1-NEXT:    __host__ __device__ bool operator()(int x) const { return x > 0; }
// thrust_not1-NEXT:    typedef int argument_type;
// thrust_not1-NEXT:  };
// thrust_not1-NEXT:  greater_than_zero pred;
// thrust_not1-NEXT:  thrust::not1(pred);
// thrust_not1-NEXT:Is migrated to:
// thrust_not1-NEXT:  struct greater_than_zero {
// thrust_not1-NEXT:    bool operator()(int x) const { return x > 0; }
// thrust_not1-NEXT:    typedef int argument_type;
// thrust_not1-NEXT:  };
// thrust_not1-NEXT:  greater_than_zero pred;
// thrust_not1-NEXT:  oneapi::dpl::not1(pred);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::not2 --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_not2
// thrust_not2:CUDA API:
// thrust_not2-NEXT:  struct greater_than_zero {
// thrust_not2-NEXT:    __host__ __device__ bool operator()(int x) const { return x > 0; }
// thrust_not2-NEXT:    typedef int argument_type;
// thrust_not2-NEXT:  };
// thrust_not2-NEXT:  greater_than_zero pred;
// thrust_not2-NEXT:  thrust::not2(thrust::greater_equal<int>());
// thrust_not2-NEXT:Is migrated to:
// thrust_not2-NEXT:  struct greater_than_zero {
// thrust_not2-NEXT:    bool operator()(int x) const { return x > 0; }
// thrust_not2-NEXT:    typedef int argument_type;
// thrust_not2-NEXT:  };
// thrust_not2-NEXT:  greater_than_zero pred;
// thrust_not2-NEXT:  std::not2(std::greater_equal<int>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::make_transform_output_iterator --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_make_transform_output_iterator
// thrust_make_transform_output_iterator:CUDA API:
// thrust_make_transform_output_iterator-NEXT:  struct square {
// thrust_make_transform_output_iterator-NEXT:    __host__ __device__ int operator()(int x) const { return x * x; }
// thrust_make_transform_output_iterator-NEXT:  };
// thrust_make_transform_output_iterator-NEXT:  const int N = 5;
// thrust_make_transform_output_iterator-NEXT:  thrust::device_vector<int> vec(N);
// thrust_make_transform_output_iterator-NEXT:  auto output_iter = thrust::make_transform_output_iterator(vec.begin(), square());
// thrust_make_transform_output_iterator-NEXT:Is migrated to:
// thrust_make_transform_output_iterator-NEXT:  struct square {
// thrust_make_transform_output_iterator-NEXT:    int operator()(int x) const { return x * x; }
// thrust_make_transform_output_iterator-NEXT:  };
// thrust_make_transform_output_iterator-NEXT:  const int N = 5;
// thrust_make_transform_output_iterator-NEXT:  dpct::device_vector<int> vec(N);
// thrust_make_transform_output_iterator-NEXT:  auto output_iter = dpct::make_transform_output_iterator(vec.begin(), square());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::device_malloc --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_device_malloc
// thrust_device_malloc:CUDA API:
// thrust_device_malloc-NEXT:  thrust::device_ptr<thrust::complex<double>> d_ptr = thrust::device_malloc<thrust::complex<double>>(1);
// thrust_device_malloc-NEXT:Is migrated to:
// thrust_device_malloc-NEXT:  dpct::device_pointer<std::complex<double>> d_ptr = dpct::malloc_device<std::complex<double>>(1);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::free --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_free
// thrust_free:CUDA API:
// thrust_free-NEXT:  const int N = 100;
// thrust_free-NEXT:  thrust::device_system_tag device_sys;
// thrust_free-NEXT:  thrust::pointer<int, thrust::device_system_tag> ptr = thrust::malloc<int>(device_sys, N);
// thrust_free-NEXT:  thrust::free(device_sys, ptr);
// thrust_free-NEXT:Is migrated to:
// thrust_free-NEXT:  const int N = 100;
// thrust_free-NEXT:  dpct::device_sys_tag device_sys;
// thrust_free-NEXT:  dpct::tagged_pointer<int, dpct::device_sys_tag> ptr = dpct::malloc<int>(device_sys, N);
// thrust_free-NEXT:  dpct::free(device_sys, ptr);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::get --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_get
// thrust_get: CUDA API:
// thrust_get-NEXT:   auto ret = thrust::make_tuple(3, 4);
// thrust_get-NEXT:  auto to = thrust::get<0>(ret);
// thrust_get-NEXT:Is migrated to:
// thrust_get-NEXT:  auto ret = std::make_tuple(3, 4);
// thrust_get-NEXT:  auto to = std::get<0>(ret);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::make_counting_iterator --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_make_counting_iterator
// thrust_make_counting_iterator:CUDA API:
// thrust_make_counting_iterator-NEXT:  auto range = thrust::make_counting_iterator(0);
// thrust_make_counting_iterator-NEXT:Is migrated to:
// thrust_make_counting_iterator-NEXT:  auto range = dpct::make_counting_iterator(0);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::make_discard_iterator --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_make_discard_iterator
// thrust_make_discard_iterator:CUDA API:
// thrust_make_discard_iterator-NEXT:  thrust::make_discard_iterator();
// thrust_make_discard_iterator-NEXT:Is migrated to:
// thrust_make_discard_iterator-NEXT:  oneapi::dpl::discard_iterator();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::make_permutation_iterator --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_make_permutation_iterator
// thrust_make_permutation_iterator:CUDA API:
// thrust_make_permutation_iterator-NEXT:  thrust::host_vector<int> h_input(10);
// thrust_make_permutation_iterator-NEXT:  thrust::host_vector<int> h_input2(10);
// thrust_make_permutation_iterator-NEXT:  thrust::make_permutation_iterator(h_input.begin(),h_input2.begin());
// thrust_make_permutation_iterator-NEXT:Is migrated to:
// thrust_make_permutation_iterator-NEXT:  std::vector<int> h_input(10);
// thrust_make_permutation_iterator-NEXT:  std::vector<int> h_input2(10);
// thrust_make_permutation_iterator-NEXT:  oneapi::dpl::make_permutation_iterator(h_input.begin(), h_input2.begin());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::make_reverse_iterator --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_make_reverse_iterator
// thrust_make_reverse_iterator:CUDA API:
// thrust_make_reverse_iterator-NEXT:  thrust::device_vector<int> d_vec(10);
// thrust_make_reverse_iterator-NEXT:  auto iter = thrust::make_reverse_iterator(d_vec.begin());
// thrust_make_reverse_iterator-NEXT:Is migrated to:
// thrust_make_reverse_iterator-NEXT:  dpct::device_vector<int> d_vec(10);
// thrust_make_reverse_iterator-NEXT:  auto iter = oneapi::dpl::make_reverse_iterator(d_vec.begin());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::make_transform_iterator --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_make_transform_iterator
// thrust_make_transform_iterator:CUDA API:
// thrust_make_transform_iterator-NEXT:  thrust::host_vector<int> h_input(10);
// thrust_make_transform_iterator-NEXT:  thrust::host_vector<int> h_input2(10);
// thrust_make_transform_iterator-NEXT:  thrust::make_transform_iterator(h_input.begin(), thrust::negate<int>());
// thrust_make_transform_iterator-NEXT:Is migrated to:
// thrust_make_transform_iterator-NEXT:  std::vector<int> h_input(10);
// thrust_make_transform_iterator-NEXT:  std::vector<int> h_input2(10);
// thrust_make_transform_iterator-NEXT:  oneapi::dpl::make_transform_iterator(h_input.begin(), std::negate<int>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::make_tuple --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_make_tuple
// thrust_make_tuple:CUDA API:
// thrust_make_tuple-NEXT:    auto ret = thrust::make_tuple(3, 4);
// thrust_make_tuple-NEXT:Is migrated to:
// thrust_make_tuple-NEXT:    auto ret = std::make_tuple(3, 4);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::make_zip_iterator --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_make_zip_iterator
// thrust_make_zip_iterator:CUDA API:
// thrust_make_zip_iterator-NEXT:  thrust::device_vector<int> int_in(3);
// thrust_make_zip_iterator-NEXT:  thrust::device_vector<float> float_in(3);
// thrust_make_zip_iterator-NEXT:  typedef thrust::device_vector<int>::iterator int_iterator;
// thrust_make_zip_iterator-NEXT:  typedef thrust::device_vector<float>::iterator float_iterator;
// thrust_make_zip_iterator-NEXT:  typedef thrust::tuple<int_iterator, float_iterator> iterator_tuple;
// thrust_make_zip_iterator-NEXT:  thrust::zip_iterator<iterator_tuple> ret = thrust::make_zip_iterator(thrust::make_tuple(int_in.begin(), float_in.begin()));
// thrust_make_zip_iterator-NEXT:Is migrated to:
// thrust_make_zip_iterator-NEXT:  dpct::device_vector<int> int_in(3);
// thrust_make_zip_iterator-NEXT:  dpct::device_vector<float> float_in(3);
// thrust_make_zip_iterator-NEXT:  typedef dpct::device_vector<int>::iterator int_iterator;
// thrust_make_zip_iterator-NEXT:  typedef dpct::device_vector<float>::iterator float_iterator;
// thrust_make_zip_iterator-NEXT:  typedef std::tuple<int_iterator, float_iterator> iterator_tuple;
// thrust_make_zip_iterator-NEXT:  dpct::zip_iterator<iterator_tuple> ret = oneapi::dpl::make_zip_iterator(std::make_tuple(int_in.begin(), float_in.begin()));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::count --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_count
// thrust_count: CUDA API:
// thrust_count-NEXT:   std::vector<int> v;
// thrust_count-NEXT:   /*1*/ thrust::count(thrust::seq, v.begin(), v.end(), 23);
// thrust_count-NEXT:   /*2*/ thrust::count(v.begin(), v.end(), 23);
// thrust_count-NEXT: Is migrated to:
// thrust_count-NEXT:   std::vector<int> v;
// thrust_count-NEXT:   /*1*/ std::count(oneapi::dpl::execution::seq, v.begin(), v.end(), 23);
// thrust_count-NEXT:   /*2*/ std::count(oneapi::dpl::execution::seq, v.begin(), v.end(), 23);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::sort --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_sort
// thrust_sort: CUDA API:
// thrust_sort-NEXT:   thrust::host_vector<int> h_vec(10);
// thrust_sort-NEXT:   thrust::device_vector<int> d_vec(10);
// thrust_sort-NEXT:   /*1*/ thrust::sort(h_vec.begin(), h_vec.end());
// thrust_sort-NEXT:   /*2*/ thrust::sort(thrust::device, d_vec.begin(), d_vec.end());
// thrust_sort-NEXT: Is migrated to:
// thrust_sort-NEXT:   std::vector<int> h_vec(10);
// thrust_sort-NEXT:   dpct::device_vector<int> d_vec(10);
// thrust_sort-NEXT:   /*1*/ oneapi::dpl::sort(oneapi::dpl::execution::seq, h_vec.begin(), h_vec.end());
// thrust_sort-NEXT:   /*2*/ oneapi::dpl::sort(oneapi::dpl::execution::make_device_policy(q_ct1), d_vec.begin(), d_vec.end());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::get_temporary_buffer --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_get_temporary_buffer
// thrust_get_temporary_buffer: CUDA API:
// thrust_get_temporary_buffer-NEXT:   const int N = 100;
// thrust_get_temporary_buffer-NEXT:   typedef thrust::pair<thrust::pointer<int, thrust::device_system_tag>, std::ptrdiff_t> ptr_and_size_t;
// thrust_get_temporary_buffer-NEXT:   thrust::device_system_tag device_sys;
// thrust_get_temporary_buffer-NEXT:   ptr_and_size_t ptr_and_size = thrust::get_temporary_buffer<int>(device_sys, N);
// thrust_get_temporary_buffer-NEXT: Is migrated to:
// thrust_get_temporary_buffer-NEXT:   const int N = 100;
// thrust_get_temporary_buffer-NEXT:   typedef std::pair<dpct::tagged_pointer<int, dpct::device_sys_tag>, std::ptrdiff_t> ptr_and_size_t;
// thrust_get_temporary_buffer-NEXT:   dpct::device_sys_tag device_sys;
// thrust_get_temporary_buffer-NEXT:   ptr_and_size_t ptr_and_size = dpct::get_temporary_allocation<int>(device_sys, N);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::max --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_max
// thrust_max: CUDA API:
// thrust_max-NEXT:   struct key_value {
// thrust_max-NEXT:     int key;
// thrust_max-NEXT:     int value;
// thrust_max-NEXT:   };
// thrust_max-NEXT:   struct compare_key_value {
// thrust_max-NEXT:     __host__ __device__ bool operator()(key_value lhs, key_value rhs) {
// thrust_max-NEXT:       return lhs.key < rhs.key;
// thrust_max-NEXT:     }
// thrust_max-NEXT:   };
// thrust_max-NEXT:   key_value a = {13, 0};
// thrust_max-NEXT:   key_value b = {7, 1};
// thrust_max-NEXT:   key_value smaller = thrust::max(a, b, compare_key_value());
// thrust_max-NEXT:   int value = thrust::max(1, 2);
// thrust_max-NEXT: Is migrated to:
// thrust_max-NEXT:   struct key_value {
// thrust_max-NEXT:     int key;
// thrust_max-NEXT:     int value;
// thrust_max-NEXT:   };
// thrust_max-NEXT:   struct compare_key_value {
// thrust_max-NEXT:     bool operator()(key_value lhs, key_value rhs) {
// thrust_max-NEXT:       return lhs.key < rhs.key;
// thrust_max-NEXT:     }
// thrust_max-NEXT:   };
// thrust_max-NEXT:   key_value a = {13, 0};
// thrust_max-NEXT:   key_value b = {7, 1};
// thrust_max-NEXT:   key_value smaller = std::max(a, b, compare_key_value());
// thrust_max-NEXT:   int value = std::max(1, 2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::min --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_min
// thrust_min: CUDA API:
// thrust_min-NEXT:   struct key_value {
// thrust_min-NEXT:     int key;
// thrust_min-NEXT:     int value;
// thrust_min-NEXT:   };
// thrust_min-NEXT:   struct compare_key_value {
// thrust_min-NEXT:     __host__ __device__ bool operator()(key_value lhs, key_value rhs) {
// thrust_min-NEXT:       return lhs.key < rhs.key;
// thrust_min-NEXT:     }
// thrust_min-NEXT:   };
// thrust_min-NEXT:   key_value a = {13, 0};
// thrust_min-NEXT:   key_value b = {7, 1};
// thrust_min-NEXT:   key_value smaller = thrust::min(a, b, compare_key_value());
// thrust_min-NEXT:   int value = thrust::min(1, 2);
// thrust_min-NEXT: Is migrated to:
// thrust_min-NEXT:   struct key_value {
// thrust_min-NEXT:     int key;
// thrust_min-NEXT:     int value;
// thrust_min-NEXT:   };
// thrust_min-NEXT:   struct compare_key_value {
// thrust_min-NEXT:     bool operator()(key_value lhs, key_value rhs) {
// thrust_min-NEXT:       return lhs.key < rhs.key;
// thrust_min-NEXT:     }
// thrust_min-NEXT:   };
// thrust_min-NEXT:   key_value a = {13, 0};
// thrust_min-NEXT:   key_value b = {7, 1};
// thrust_min-NEXT:   key_value smaller = std::min(a, b, compare_key_value());
// thrust_min-NEXT:   int value = std::min(1, 2);
