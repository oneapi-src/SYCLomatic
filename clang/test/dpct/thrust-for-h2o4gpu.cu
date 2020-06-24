// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/thrust-for-h2o4gpu.dp.cpp --match-full-lines %s


// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <dpct/dpstd_utils.hpp>
// CHECK-NEXT: #include <dpstd/execution>
// CHECK-NEXT: #include <dpstd/algorithm>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <algorithm>
#include <thrust/inner_product.h>
#include <thrust/extrema.h>


template <typename T> struct is_even {
  __host__ __device__ bool operator()(T x) {
    return (static_cast<unsigned int>(x) & 1) == 0;
  }
};

template <typename T> struct absolute_value {
  __host__ __device__ void operator()(T &x) const { x = (x > 0 ? x : -x); }
};

//CHECK: template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Predicate, typename Iterator3>
//CHECK-NEXT: void copy_if_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result1, Predicate pred, Iterator3 result2)
//CHECK-NEXT: {
//CHECK-NEXT:   *result2 = std::copy_if(exec, first, last, result1, pred);
//CHECK-NEXT: }
template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Predicate, typename Iterator3>
__global__ void copy_if_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result1, Predicate pred, Iterator3 result2)
{
  *result2 = thrust::copy_if(exec, first, last, result1, pred);
}

template<typename ExecutionPolicy>
void copy_if_device(ExecutionPolicy exec)
{
  size_t n = 1000;

  //CHECK: std::vector<int>   h_data (n, 1);
  //CHECK-NEXT: dpct::device_vector<int> d_data = h_data;
  //CHECK-NEXT: typename dpct::device_vector<int>::iterator d_new_end;
  //CHECK-NEXT: dpct::device_vector<int> d_result(n);
  //CHECK-NEXT: dpct::device_vector<typename dpct::device_vector<int>::iterator> dd(1);
  thrust::host_vector<int>   h_data (n, 1);
  thrust::device_vector<int> d_data = h_data;
  typename thrust::device_vector<int>::iterator d_new_end;
  thrust::device_vector<int> d_result(n);
  thrust::device_vector<typename thrust::device_vector<int>::iterator> dd(1);

  //CHECK: dpct::get_default_queue().submit(
  //CHECK-NEXT:  [&](sycl::handler &cgh) {
  //CHECK-NEXT:    auto d_data_begin_ct1 = d_data.begin();
  //CHECK-NEXT:    auto d_data_end_ct2 = d_data.end();
  //CHECK-NEXT:    auto d_result_begin_ct3 = d_result.begin();
  //CHECK-NEXT:    auto dd_begin_ct5 = dd.begin();
  //CHECK-EMPTY:
  //CHECK-NEXT:    cgh.parallel_for(
  //CHECK-NEXT:      sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  //CHECK-NEXT:      [=](sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:        copy_if_kernel(exec, d_data_begin_ct1, d_data_end_ct2, d_result_begin_ct3, is_even<int>(), dd_begin_ct5);
  //CHECK-NEXT:      });
  //CHECK-NEXT:  });
  copy_if_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), d_result.begin(), is_even<int>(), dd.begin());
}

template<typename T>
struct isnan_test {
    __host__ __device__ bool operator()(const T a) const {
        return isnan(a) || isinf(a);
    }
};

void foo() {
  //CHECK: copy_if_device(dpstd::execution::seq);
  copy_if_device(thrust::seq);

  //CHECK: std::vector<int> h_data(10, 1);
  //CHECK-NEXT: std::vector<int> h_result(10);
  //CHECK-NEXT: dpct::device_vector<int> *data[10];
  //CHECK-NEXT: dpct::device_vector<int> d_new_potential_centroids(10);
  //CHECK-NEXT: auto range = dpct::make_counting_iterator(0);
  thrust::host_vector<int> h_data(10, 1);
  thrust::host_vector<int> h_result(10);
  thrust::device_vector<int> *data[10];
  thrust::device_vector<int> d_new_potential_centroids(10);
  auto range = thrust::make_counting_iterator(0);

  //CHECK: std::copy_if(dpstd::execution::make_device_policy(q_ct1), h_data.begin(), h_data.end(), h_result.begin(), is_even<int>());
  //CHECK-NEXT: std::copy_if(dpstd::execution::seq, h_data.begin(), h_data.end(), h_result.begin(), is_even<int>());
  //CHECK-NEXT: dpct::copy_if(dpstd::execution::make_device_policy(q_ct1), (*data[0]).begin(), (*data[0]).end(), range, d_new_potential_centroids.begin(),[=] (int idx) { return true; });
  thrust::copy_if(h_data.begin(), h_data.end(), h_result.begin(), is_even<int>());
  thrust::copy_if(thrust::seq, h_data.begin(), h_data.end(), h_result.begin(), is_even<int>());
  thrust::copy_if((*data[0]).begin(), (*data[0]).end(), range, d_new_potential_centroids.begin(),[=] __device__(int idx) { return true; });

  //CHECK: std::vector<dpct::device_vector<int>> d(10);
  //CHECK-NEXT: auto t = dpct::make_counting_iterator(0);
  //CHECK-NEXT: auto min_costs_ptr = dpct::raw_pointer_cast(d[0].data());
  //CHECK-NEXT: int pot_cent_num = std::count_if(dpstd::execution::make_device_policy(q_ct1), t, t + 10, [=] (int idx) { return true;});
  std::vector<thrust::device_vector<int>> d(10);
  auto t = thrust::make_counting_iterator(0);
  auto min_costs_ptr = thrust::raw_pointer_cast(d[0].data());
  int pot_cent_num = thrust::count_if(t, t + 10, [=] __device__(int idx) { return true;});

  {
  float *_de = NULL;
  float fill_value = 0.0;

  //CHECK: dpct::device_ptr<float> dev_ptr = dpct::device_pointer_cast(static_cast<float *>(&_de[0]));
  //CHECK-NEXT: std::fill(dpstd::execution::make_device_policy(q_ct1), dev_ptr, dev_ptr + 10, fill_value);
  //CHECK-NEXT: std::fill_n(dpstd::execution::make_device_policy(q_ct1), dev_ptr, 10, fill_value);
  //CHECK-NEXT: float M_inner = dpct::inner_product(dpstd::execution::make_device_policy(q_ct1), dev_ptr, dev_ptr + 10, dev_ptr, 0.0f);
  thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(static_cast<float *>(&_de[0]));
  thrust::fill(dev_ptr, dev_ptr + 10, fill_value);
  thrust::fill_n(dev_ptr, 10, fill_value);
  float M_inner = thrust::inner_product(dev_ptr, dev_ptr + 10, dev_ptr, 0.0f);
  }

 {
  //CHECK: dpct::device_vector<double> t;
  //CHECK-NEXT: std::for_each( dpstd::execution::make_device_policy(q_ct1), t.begin(), t.end(), absolute_value<double>());
  thrust::device_vector<double> t;
  thrust::for_each( t.begin(), t.end(), absolute_value<double>());
 }

 {
  //CHECK: int min = std::min(1, 2);
  //CHECK-NEXT: int max = std::max(1, 2);
  int min = thrust::min(1, 2);
  int max = thrust::max(1, 2);
 }

 {
  //CHECK: dpct::device_vector<int> a, b, c;
  //CHECK-NEXT: dpct::sort_by_key(dpstd::execution::make_device_policy(q_ct1), a.begin(), b.end(), c.begin());
  thrust::device_vector<int> a, b, c;
  thrust::sort_by_key(a.begin(), b.end(), c.begin());
 }

 {
  const int N = 1000;
  //CHECK: dpct::device_vector<float> t1(N);
  //CHECK-NEXT: dpct::device_vector<float> t2(N);
  //CHECK-NEXT: dpct::device_vector<float> t3(N);
  //CHECK-NEXT: std::transform(dpstd::execution::make_device_policy(q_ct1), t1.begin(), t1.end(), t2.begin(), t3.begin(), std::divides<float>());
  //CHECK-NEXT: std::transform(dpstd::execution::make_device_policy(q_ct1), t1.begin(), t1.end(), t2.begin(), t3.begin(), std::multiplies<float>());
  //CHECK-NEXT: std::transform(dpstd::execution::make_device_policy(q_ct1), t1.begin(), t1.end(), t2.begin(), t3.begin(), std::plus<float>());
  thrust::device_vector<float> t1(N);
  thrust::device_vector<float> t2(N);
  thrust::device_vector<float> t3(N);
  thrust::transform(t1.begin(), t1.end(), t2.begin(), t3.begin(), thrust::divides<float>());
  thrust::transform(t1.begin(), t1.end(), t2.begin(), t3.begin(), thrust::multiplies<float>());
  thrust::transform(t1.begin(), t1.end(), t2.begin(), t3.begin(), thrust::plus<float>());
 }

 {
    //CHECK: dpct::device_vector<int> data(4);
    //CHECK-NEXT: std::transform(dpstd::execution::make_device_policy(q_ct1), data.begin(), data.end(), dpct::make_constant_iterator(10), data.begin(), std::divides<int>());
    thrust::device_vector<int> data(4);
    thrust::transform(data.begin(), data.end(), thrust::make_constant_iterator(10), data.begin(), thrust::divides<int>());
 }

 {
    //CHECK: std::tuple<int, const char *> t(13, "foo");
    //CHECK-NEXT: std::cout << "The 1st value of t is " << std::get<0>(t) << std::endl;
    //CHECK-NEXT: auto ret = std::make_tuple(3, 4);
    thrust::tuple<int, const char *> t(13, "foo");
    std::cout << "The 1st value of t is " << thrust::get<0>(t) << std::endl;
    auto ret = thrust::make_tuple(3, 4);
 }

 {
  //CHECK: dpct::device_vector<int> int_in(3);
  //CHECK-NEXT: dpct::device_vector<float> float_in(3);
  //CHECK-NEXT: auto ret = dpstd::make_zip_iterator(int_in.begin(), float_in.begin());
  //CHECK-NEXT: auto arg = std::make_tuple(int_in.begin(), float_in.begin());
  //CHECK-NEXT: auto ret_1 = dpstd::make_zip_iterator(std::get<0>(arg), std::get<1>(arg));
  thrust::device_vector<int> int_in(3);
  thrust::device_vector<float> float_in(3);
  auto ret = thrust::make_zip_iterator(thrust::make_tuple(int_in.begin(), float_in.begin()));
  auto arg = thrust::make_tuple(int_in.begin(), float_in.begin());
  auto ret_1 = thrust::make_zip_iterator(arg);
 }

 {
  //CHECK: int x =  137;
  //CHECK-NEXT: int y = -137;
  //CHECK-NEXT: dpstd::maximum<int> mx;
  //CHECK-NEXT: int value = mx(x,y);
  int x =  137;
  int y = -137;
  thrust::maximum<int> mx;
  int value = mx(x,y);
 }

 {
  //CHECK: dpct::device_ptr<int> begin;
  //CHECK-NEXT: dpct::device_ptr<int> end;
  //CHECK-NEXT: bool h_result = std::transform_reduce(dpstd::execution::make_device_policy(q_ct1), begin, end, 0, std::plus<bool>(), isnan_test<int>());
  //CHECK-NEXT: bool h_result_1 = std::transform_reduce(dpstd::execution::seq, begin, end, 0, std::plus<bool>(), isnan_test<int>());
  thrust::device_ptr<int> begin;
  thrust::device_ptr<int> end;
  bool h_result = thrust::transform_reduce(begin, end, isnan_test<int>(), 0, thrust::plus<bool>());
  bool h_result_1 = thrust::transform_reduce(thrust::seq, begin, end, isnan_test<int>(), 0, thrust::plus<bool>());

 }
}
