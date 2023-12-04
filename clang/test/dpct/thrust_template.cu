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
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/unique.h>
#include <thrust/sort.h>
#include <thrust/iterator/constant_iterator.h>

// for cuda 12.0
#include <thrust/execution_policy.h>

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

template <class T>
void foo_host(){
  thrust::device_vector<T> A(4);
  A[0] = -5;
  A[1] = 3;
  A[2] = 0;
  A[3] = 4;
  thrust::device_vector<T> S(4);
  S[0] = -1;
  S[1] =  0;
  S[2] = -1;
  S[3] =  1;
  thrust::device_vector<T> R(4);

  std::vector<T> B(4);
  B[0] = -5;
  B[1] = 3;
  B[2] = 0;
  B[3] = 4;
  std::vector<T> S2(4);
  S2[0] = -1;
  S2[1] =  0;
  S2[2] = -1;
  S2[3] =  1;
  std::vector<T> R2(4);

  greater_than_zero pred;

  //CHECK: oneapi::dpl::equal_to<T>();
  thrust::equal_to<T>();
  //CHECK: oneapi::dpl::less<T>();
  thrust::less<T>();
  //CHECK: oneapi::dpl::not1(pred);
  thrust::not1(pred);

  //CHECK: oneapi::dpl::replace_if(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), pred, 0);
  //CHECK-NEXT: oneapi::dpl::replace_if(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), pred, 0);
  //CHECK-NEXT: dpct::replace_if(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), S.begin(), pred, 0);
  //CHECK-NEXT: dpct::replace_if(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), S.begin(), pred, 0);
  //CHECK-NEXT: oneapi::dpl::replace_if(oneapi::dpl::execution::seq, B.begin(), B.end(), pred, 0);
  //CHECK-NEXT: dpct::replace_if(oneapi::dpl::execution::seq, B.begin(), B.end(), S2.begin(), pred, 0);
  thrust::replace_if(thrust::device, A.begin(), A.end(), pred, 0);
  thrust::replace_if(A.begin(), A.end(), pred, 0);
  thrust::replace_if(thrust::device, A.begin(), A.end(), S.begin(), pred, 0);
  thrust::replace_if(A.begin(), A.end(), S.begin(), pred, 0);
  thrust::replace_if(thrust::seq, B.begin(), B.end(), pred, 0);
  thrust::replace_if(thrust::seq, B.begin(), B.end(), S2.begin(), pred, 0);

  //CHECK: oneapi::dpl::remove_if(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), pred);
  //CHECK-NEXT: oneapi::dpl::remove_if(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), pred);
  //CHECK-NEXT: dpct::remove_if(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), S.begin(), pred);
  //CHECK-NEXT: dpct::remove_if(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), S.begin(), pred);
  //CHECK-NEXT: oneapi::dpl::remove_if(oneapi::dpl::execution::seq, B.begin(), B.end(), pred);
  //CHECK-NEXT: dpct::remove_if(oneapi::dpl::execution::seq, B.begin(), B.end(), S2.begin(), pred);
  thrust::remove_if(thrust::device, A.begin(), A.end(), pred);
  thrust::remove_if(A.begin(), A.end(), pred);
  thrust::remove_if(thrust::device, A.begin(), A.end(), S.begin(), pred);
  thrust::remove_if(A.begin(), A.end(), S.begin(), pred);
  thrust::remove_if(thrust::seq, B.begin(), B.end(), pred);
  thrust::remove_if(thrust::seq, B.begin(), B.end(), S2.begin(), pred);


  //CHECK: oneapi::dpl::remove_copy_if(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), R.begin(), pred);
  //CHECK-NEXT: oneapi::dpl::remove_copy_if(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), R.begin(), pred);
  //CHECK-NEXT: dpct::remove_copy_if(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), S.begin(), R.begin(), pred);
  //CHECK-NEXT: dpct::remove_copy_if(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), S.begin(), R.begin(), pred);
  //CHECK-NEXT: oneapi::dpl::remove_copy_if(oneapi::dpl::execution::seq, B.begin(), B.end(), R2.begin(), pred);
  //CHECK-NEXT: dpct::remove_copy_if(oneapi::dpl::execution::seq, B.begin(), B.end(), S2.begin(), R2.begin(), pred);
  thrust::remove_copy_if(thrust::device, A.begin(), A.end(), R.begin(), pred);
  thrust::remove_copy_if(A.begin(), A.end(), R.begin(), pred);
  thrust::remove_copy_if(thrust::device, A.begin(), A.end(), S.begin(), R.begin(), pred);
  thrust::remove_copy_if(A.begin(), A.end(), S.begin(), R.begin(), pred);
  thrust::remove_copy_if(thrust::seq, B.begin(), B.end(), R2.begin(), pred);
  thrust::remove_copy_if(thrust::seq, B.begin(), B.end(), S2.begin(), R2.begin(), pred);

  //CHECK: oneapi::dpl::any_of(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), pred);
  //CHECK-NEXT: oneapi::dpl::any_of(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), pred);
  //CHECK-NEXT: oneapi::dpl::any_of(oneapi::dpl::execution::make_device_policy(q_ct1), B.begin(), B.end(), pred);
  thrust::any_of(A.begin(), A.end(), pred);
  thrust::any_of(thrust::device, A.begin(), A.end(), pred);
  thrust::any_of(thrust::device, B.begin(), B.end(), pred);

  //CHECK: oneapi::dpl::replace(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), 0, 399);
  //CHECK-NEXT: oneapi::dpl::replace(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), 0, 399);
  //CHECK-NEXT: oneapi::dpl::replace(oneapi::dpl::execution::make_device_policy(q_ct1), B.begin(), B.end(), 0, 399);
  thrust::replace(A.begin(), A.end(), 0, 399);
  thrust::replace(thrust::device, A.begin(), A.end(), 0, 399);
  thrust::replace(thrust::device, B.begin(), B.end(), 0, 399);

  //CHECK: #define TM std::minus<T>()
  //CHECK-NEXT: oneapi::dpl::adjacent_difference(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), R.begin(), TM);
  //CHECK-NEXT: oneapi::dpl::adjacent_difference(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), R.begin(), std::minus<T>());
  //CHECK-NEXT: oneapi::dpl::adjacent_difference(oneapi::dpl::execution::seq, B.begin(), B.end(), R2.begin(), std::minus<T>());
  //CHECK-NEXT: oneapi::dpl::adjacent_difference(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), R.begin());
  //CHECK-NEXT: oneapi::dpl::adjacent_difference(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), R.begin());
  //CHECK-NEXT: oneapi::dpl::adjacent_difference(oneapi::dpl::execution::seq, B.begin(), B.end(), R2.begin());
  #define TM thrust::minus<T>()
  thrust::adjacent_difference(A.begin(), A.end(), R.begin(), TM);
  thrust::adjacent_difference(thrust::device, A.begin(), A.end(), R.begin(), thrust::minus<T>());
  thrust::adjacent_difference(thrust::seq, B.begin(), B.end(), R2.begin(), thrust::minus<T>());
  thrust::adjacent_difference(A.begin(), A.end(), R.begin());
  thrust::adjacent_difference(thrust::device, A.begin(), A.end(), R.begin());
  thrust::adjacent_difference(thrust::seq, B.begin(), B.end(), R2.begin());

  //CHECK: oneapi::dpl::inclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), R.begin(), TM);
  //CHECK-NEXT: oneapi::dpl::inclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), R.begin(), std::minus<T>());
  //CHECK-NEXT: oneapi::dpl::inclusive_scan(oneapi::dpl::execution::seq, B.begin(), B.end(), R2.begin(), std::minus<T>());
  //CHECK-NEXT: oneapi::dpl::inclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), R.begin());
  //CHECK-NEXT: oneapi::dpl::inclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), R.begin());
  //CHECK-NEXT: oneapi::dpl::inclusive_scan(oneapi::dpl::execution::seq, B.begin(), B.end(), R2.begin());
  thrust::inclusive_scan(A.begin(), A.end(), R.begin(), TM);
  thrust::inclusive_scan(thrust::device, A.begin(), A.end(), R.begin(), thrust::minus<T>());
  thrust::inclusive_scan(thrust::seq, B.begin(), B.end(), R2.begin(), thrust::minus<T>());
  thrust::inclusive_scan(A.begin(), A.end(), R.begin());
  thrust::inclusive_scan(thrust::device, A.begin(), A.end(), R.begin());
  thrust::inclusive_scan(thrust::seq, B.begin(), B.end(), R2.begin());
}

struct s_pred_A {
  __host__ __device__ bool operator()(short int x) const { return true; }
  typedef short int argument_type;
};
static s_pred_A pred_A;

template <typename ELT_TYPE> void testfunc() {
  thrust::host_vector<ELT_TYPE> V1(1);
  thrust::host_vector<short int> V2(1);
  thrust::host_vector<short int> V6(1);
  V1[0x0] = 0x0;
  V2[0x0] = 0x0;
  V6[0x0] = 0xfcdd;

  // CHECK: dpct::remove_copy_if(oneapi::dpl::execution::seq, V2.begin(), V2.end(), V1.begin(), V6.begin(), pred_A);
  thrust::remove_copy_if(V2.begin(), V2.end(), V1.begin(), V6.begin(), pred_A);

  // CHECK: oneapi::dpl::unique_copy(oneapi::dpl::execution::seq, V2.begin(), V2.end(), V2.begin());
  thrust::unique_copy(V2.begin(), V2.end(), V2.begin());

  // CHECK:oneapi::dpl::stable_sort(oneapi::dpl::execution::seq, V1.begin(), V1.end(), std::not2(std::greater_equal<int>()));
  thrust::stable_sort(V1.begin(),V1.end(),thrust::not2(thrust::greater_equal<int>()));
}

int main() {
  testfunc<short int>();
  return 0;
}


template <typename Iterator>
void foo2() {
  // CHECK: typedef typename std::tuple_element_t<0, typename Iterator::value_type> Type;
  typedef typename Iterator::value_type :: head_type Type;
}

void foo3() {
  foo2<thrust::constant_iterator<thrust::tuple<float, double>>>();
}
