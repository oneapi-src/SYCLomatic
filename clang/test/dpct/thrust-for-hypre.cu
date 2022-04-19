// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: c2s -out-root %T/thrust-for-hypre %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17 -fsized-deallocation
// RUN: FileCheck --input-file %T/thrust-for-hypre/thrust-for-hypre.dp.cpp --match-full-lines %s
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <thrust/adjacent_difference.h>

struct greater_than_zero
{
  __host__ __device__
  bool operator()(int x) const
  {
    return x > 0;
  }
  typedef int argument_type;
};

template<class T1, class T2>
void foo2(T1 policy, T2 vec){
  thrust::device_vector<int> R(4);
  //CHECK: oneapi::dpl::inclusive_scan(oneapi::dpl::execution::make_device_policy(c2s::get_default_queue()), vec.begin(), vec.end(), R.begin(), std::minus<int>());
  thrust::inclusive_scan(policy, vec.begin(), vec.end(), R.begin(), thrust::minus<int>());
}

int main(){
    return 0;
}

void foo_host(){
  thrust::device_vector<int> A(4);
  A[0] = -5;
  A[1] = 3;
  A[2] = 0;
  A[3] = 4;
  thrust::device_vector<int> S(4);
  S[0] = -1;
  S[1] =  0;
  S[2] = -1;
  S[3] =  1;
  thrust::device_vector<int> R(4);

  std::vector<int> B(4);
  B[0] = -5;
  B[1] = 3;
  B[2] = 0;
  B[3] = 4;
  std::vector<int> S2(4);
  S2[0] = -1;
  S2[1] =  0;
  S2[2] = -1;
  S2[3] =  1;
  std::vector<int> R2(4);

  greater_than_zero pred;

  //CHECK: oneapi::dpl::equal_to<int>();
  thrust::equal_to<int>();
  //CHECK: oneapi::dpl::less<int>();
  thrust::less<int>();
  //CHECK: oneapi::dpl::not1(pred);
  thrust::not1(pred);

  //CHECK: oneapi::dpl::replace_if(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), pred, 0);
  //CHECK-NEXT: oneapi::dpl::replace_if(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), pred, 0);
  //CHECK-NEXT: c2s::replace_if(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), S.begin(), pred, 0);
  //CHECK-NEXT: c2s::replace_if(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), S.begin(), pred, 0);
  //CHECK-NEXT: oneapi::dpl::replace_if(oneapi::dpl::execution::seq, B.begin(), B.end(), pred, 0);
  //CHECK-NEXT: oneapi::dpl::replace_if(oneapi::dpl::execution::seq, B.begin(), B.end(), pred, 0);
  //CHECK-NEXT: c2s::replace_if(oneapi::dpl::execution::seq, B.begin(), B.end(), S2.begin(), pred, 0);
  //CHECK-NEXT: c2s::replace_if(oneapi::dpl::execution::seq, B.begin(), B.end(), S2.begin(), pred, 0);
  thrust::replace_if(thrust::device, A.begin(), A.end(), pred, 0);
  thrust::replace_if(A.begin(), A.end(), pred, 0);
  thrust::replace_if(thrust::device, A.begin(), A.end(), S.begin(), pred, 0);
  thrust::replace_if(A.begin(), A.end(), S.begin(), pred, 0);
  thrust::replace_if(thrust::seq, B.begin(), B.end(), pred, 0);
  thrust::replace_if(B.begin(), B.end(), pred, 0);
  thrust::replace_if(thrust::seq, B.begin(), B.end(), S2.begin(), pred, 0);
  thrust::replace_if(B.begin(), B.end(), S2.begin(), pred, 0);

  //CHECK: oneapi::dpl::remove_if(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), pred);
  //CHECK-NEXT: oneapi::dpl::remove_if(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), pred);
  //CHECK-NEXT: c2s::remove_if(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), S.begin(), pred);
  //CHECK-NEXT: c2s::remove_if(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), S.begin(), pred);
  //CHECK-NEXT: oneapi::dpl::remove_if(oneapi::dpl::execution::seq, B.begin(), B.end(), pred);
  //CHECK-NEXT: oneapi::dpl::remove_if(oneapi::dpl::execution::seq, B.begin(), B.end(), pred);
  //CHECK-NEXT: c2s::remove_if(oneapi::dpl::execution::seq, B.begin(), B.end(), S2.begin(), pred);
  //CHECK-NEXT: c2s::remove_if(oneapi::dpl::execution::seq, B.begin(), B.end(), S2.begin(), pred);
  thrust::remove_if(thrust::device, A.begin(), A.end(), pred);
  thrust::remove_if(A.begin(), A.end(), pred);
  thrust::remove_if(thrust::device, A.begin(), A.end(), S.begin(), pred);
  thrust::remove_if(A.begin(), A.end(), S.begin(), pred);
  thrust::remove_if(thrust::seq, B.begin(), B.end(), pred);
  thrust::remove_if(B.begin(), B.end(), pred);
  thrust::remove_if(thrust::seq, B.begin(), B.end(), S2.begin(), pred);
  thrust::remove_if(B.begin(), B.end(), S2.begin(), pred);


  //CHECK: oneapi::dpl::remove_copy_if(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), R.begin(), pred);
  //CHECK-NEXT: oneapi::dpl::remove_copy_if(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), R.begin(), pred);
  //CHECK-NEXT: c2s::remove_copy_if(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), S.begin(), R.begin(), pred);
  //CHECK-NEXT: c2s::remove_copy_if(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), S.begin(), R.begin(), pred);
  //CHECK-NEXT: oneapi::dpl::remove_copy_if(oneapi::dpl::execution::seq, B.begin(), B.end(), R2.begin(), pred);
  //CHECK-NEXT: oneapi::dpl::remove_copy_if(oneapi::dpl::execution::seq, B.begin(), B.end(), R2.begin(), pred);
  //CHECK-NEXT: c2s::remove_copy_if(oneapi::dpl::execution::seq, B.begin(), B.end(), S2.begin(), R2.begin(), pred);
  //CHECK-NEXT: c2s::remove_copy_if(oneapi::dpl::execution::seq, B.begin(), B.end(), S2.begin(), R2.begin(), pred);
  thrust::remove_copy_if(thrust::device, A.begin(), A.end(), R.begin(), pred);
  thrust::remove_copy_if(A.begin(), A.end(), R.begin(), pred);
  thrust::remove_copy_if(thrust::device, A.begin(), A.end(), S.begin(), R.begin(), pred);
  thrust::remove_copy_if(A.begin(), A.end(), S.begin(), R.begin(), pred);
  thrust::remove_copy_if(thrust::seq, B.begin(), B.end(), R2.begin(), pred);
  thrust::remove_copy_if(B.begin(), B.end(), R2.begin(), pred);
  thrust::remove_copy_if(thrust::seq, B.begin(), B.end(), S2.begin(), R2.begin(), pred);
  thrust::remove_copy_if(B.begin(), B.end(), S2.begin(), R2.begin(), pred);

  //CHECK: oneapi::dpl::any_of(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), pred);
  //CHECK-NEXT: oneapi::dpl::any_of(oneapi::dpl::execution::seq, B.begin(), B.end(), pred);
  //CHECK-NEXT: oneapi::dpl::any_of(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), pred);
  //CHECK-NEXT: oneapi::dpl::any_of(oneapi::dpl::execution::make_device_policy(q_ct1), B.begin(), B.end(), pred);
  thrust::any_of(A.begin(), A.end(), pred);
  thrust::any_of(B.begin(), B.end(), pred);
  thrust::any_of(thrust::device, A.begin(), A.end(), pred);
  thrust::any_of(thrust::device, B.begin(), B.end(), pred);

  //CHECK: oneapi::dpl::replace(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), 0, 399);
  //CHECK-NEXT: oneapi::dpl::replace(oneapi::dpl::execution::seq, B.begin(), B.end(), 0, 399);
  //CHECK-NEXT: oneapi::dpl::replace(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), 0, 399);
  //CHECK-NEXT: oneapi::dpl::replace(oneapi::dpl::execution::make_device_policy(q_ct1), B.begin(), B.end(), 0, 399);
  thrust::replace(A.begin(), A.end(), 0, 399);
  thrust::replace(B.begin(), B.end(), 0, 399);
  thrust::replace(thrust::device, A.begin(), A.end(), 0, 399);
  thrust::replace(thrust::device, B.begin(), B.end(), 0, 399);

  //CHECK: #define TM std::minus<int>()
  //CHECK-NEXT: oneapi::dpl::adjacent_difference(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), R.begin(), TM);
  //CHECK-NEXT: oneapi::dpl::adjacent_difference(oneapi::dpl::execution::seq, B.begin(), B.end(), R2.begin(), std::minus<int>());
  //CHECK-NEXT: oneapi::dpl::adjacent_difference(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), R.begin(), std::minus<int>());
  //CHECK-NEXT: oneapi::dpl::adjacent_difference(oneapi::dpl::execution::seq, B.begin(), B.end(), R2.begin(), std::minus<int>());
  //CHECK-NEXT: oneapi::dpl::adjacent_difference(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), R.begin());
  //CHECK-NEXT: oneapi::dpl::adjacent_difference(oneapi::dpl::execution::seq, B.begin(), B.end(), R2.begin());
  //CHECK-NEXT: oneapi::dpl::adjacent_difference(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), R.begin());
  //CHECK-NEXT: oneapi::dpl::adjacent_difference(oneapi::dpl::execution::seq, B.begin(), B.end(), R2.begin());
  #define TM thrust::minus<int>()
  thrust::adjacent_difference(A.begin(), A.end(), R.begin(), TM);
  thrust::adjacent_difference(B.begin(), B.end(), R2.begin(), thrust::minus<int>());
  thrust::adjacent_difference(thrust::device, A.begin(), A.end(), R.begin(), thrust::minus<int>());
  thrust::adjacent_difference(thrust::seq, B.begin(), B.end(), R2.begin(), thrust::minus<int>());
  thrust::adjacent_difference(A.begin(), A.end(), R.begin());
  thrust::adjacent_difference(B.begin(), B.end(), R2.begin());
  thrust::adjacent_difference(thrust::device, A.begin(), A.end(), R.begin());
  thrust::adjacent_difference(thrust::seq, B.begin(), B.end(), R2.begin());

  //CHECK: oneapi::dpl::inclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), R.begin(), TM);
  //CHECK-NEXT: oneapi::dpl::inclusive_scan(oneapi::dpl::execution::seq, B.begin(), B.end(), R2.begin(), std::minus<int>());
  //CHECK-NEXT: oneapi::dpl::inclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), R.begin(), std::minus<int>());
  //CHECK-NEXT: oneapi::dpl::inclusive_scan(oneapi::dpl::execution::seq, B.begin(), B.end(), R2.begin(), std::minus<int>());
  //CHECK-NEXT: oneapi::dpl::inclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), R.begin());
  //CHECK-NEXT: oneapi::dpl::inclusive_scan(oneapi::dpl::execution::seq, B.begin(), B.end(), R2.begin());
  //CHECK-NEXT: oneapi::dpl::inclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), A.begin(), A.end(), R.begin());
  //CHECK-NEXT: oneapi::dpl::inclusive_scan(oneapi::dpl::execution::seq, B.begin(), B.end(), R2.begin());
  //CHECK-NEXT: foo2(oneapi::dpl::execution::make_device_policy(q_ct1), A);
  thrust::inclusive_scan(A.begin(), A.end(), R.begin(), TM);
  thrust::inclusive_scan(B.begin(), B.end(), R2.begin(), thrust::minus<int>());
  thrust::inclusive_scan(thrust::device, A.begin(), A.end(), R.begin(), thrust::minus<int>());
  thrust::inclusive_scan(thrust::seq, B.begin(), B.end(), R2.begin(), thrust::minus<int>());
  thrust::inclusive_scan(A.begin(), A.end(), R.begin());
  thrust::inclusive_scan(B.begin(), B.end(), R2.begin());
  thrust::inclusive_scan(thrust::device, A.begin(), A.end(), R.begin());
  thrust::inclusive_scan(thrust::seq, B.begin(), B.end(), R2.begin());
  foo2(thrust::device, A);
}

