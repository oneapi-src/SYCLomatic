// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct -out-root %T/thrust_template %s --cuda-include-path="%cuda-path/include" --extra-arg="-fno-delayed-template-parsing"
// RUN: FileCheck --input-file %T/thrust_template/thrust_template.dp.cpp --match-full-lines %s

#include <thrust/device_vector.h>
#include <thrust/replace.h>
#include <thrust/scan.h>
#include <thrust/adjacent_difference.h>
#include <thrust/remove.h>

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
// So check argument number first to avoid accessing un-exsisting arg.
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
