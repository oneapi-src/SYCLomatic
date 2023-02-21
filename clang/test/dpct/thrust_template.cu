// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct -out-root %T/thrust_template %s --cuda-include-path="%cuda-path/include" --extra-arg="-fno-delayed-template-parsing" -- -ferror-limit=50
// RUN: FileCheck --input-file %T/thrust_template/thrust_template.dp.cpp --match-full-lines %s

#include <thrust/device_vector.h>
#include <thrust/replace.h>
#include <thrust/scan.h>
#include <thrust/adjacent_difference.h>
#include <thrust/remove.h>
#include <thrust/tuple.h>
#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>
#include <thrust/complex.h>
#include <thrust/iterator/permutation_iterator.h>

struct greater_than_zero
{
  __host__ __device__
  bool operator()(int x) const
  {
    return x > 0;
  }
  typedef int argument_type;
};

// Test description:
// This test is to cover thrust functions in template function.
// The direct callee of those thrust functions cannot be found when comparing the arg type.
// So check argument number first to avoid accessing un-existing arg.
template <class T>
void foo() {
  greater_than_zero pred;
  thrust::device_vector<T> A(4);
  //CHECK:oneapi::dpl::replace_if(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), pred, 0);
  thrust::replace_if(A.begin(), A.end(), pred, 0);
  T *offset = (T *)malloc(sizeof(T));
  //CHECK:oneapi::dpl::inclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), A.begin());
  thrust::inclusive_scan(A.begin(), A.end(), A.begin());
  //CHECK:oneapi::dpl::adjacent_difference(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), A.begin());
  thrust::adjacent_difference(A.begin(), A.end(), A.begin());
  //CHECK:oneapi::dpl::remove_if(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), pred);
  thrust::remove_if(A.begin(), A.end(), pred);
}

// Test description:
// This test is to check migration of thust API template arguments in non-instantiated class template
template <int current, typename I, typename F> struct misc_helper {
  template <int num, typename... Args>
  __host__ __device__ static void bar(thrust::tuple<Args...> ret) {
    {
      //CHECK:auto to = std::get<current>(ret);
      auto to = thrust::get<current>(ret);
      //CHECK:to = std::get<0>(ret);
      to = thrust::get<0>(ret);
    }

    {
      //CHECK:auto less = oneapi::dpl::less<int>();
      auto less = thrust::less<int>();
      //CHECK:oneapi::dpl::equal_to<float> pred_eq;
      thrust::equal_to<float> pred_eq;
      //CHECK: dpct::device_vector<F> vec(4);
      thrust::device_vector<F> vec(4);
      //CHECK:auto d_ptr = dpct::malloc_device<std::complex<double>>(1);
      auto d_ptr = thrust::device_malloc<thrust::complex<double>>(1);
      float *a;
      //CHECK:dpct::get_device_pointer<float>(a);
      thrust::device_pointer_cast<float>(a);
      //CHECK:oneapi::dpl::identity();
      thrust::identity<int>();
      //CHECK:oneapi::dpl::permutation_iterator<I, F> pIt;
      thrust::permutation_iterator<I, F> pIt;
    }

    {
      //CHECK:using RetTy_0 = typename std::tuple_element<0, std::tuple<Args...>>::type;
      using RetTy_0 = typename thrust::tuple_element<0, thrust::tuple<Args...>>::type;
      //CHECK:typename std::tuple_element<1, std::tuple<Args...>>::type RetTy_1;
      typename thrust::tuple_element<1, thrust::tuple<Args...>>::type RetTy_1;
      //CHECK:typedef typename std::tuple_element<2, std::tuple<Args...>>::type RetTy_2;
      typedef typename thrust::tuple_element<2, thrust::tuple<Args...>>::type RetTy_2;
    }
  }
};