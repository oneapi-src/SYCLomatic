// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct -out-root %T/thrust-for-hypre %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17
// RUN: FileCheck --input-file %T/thrust-for-hypre/thrust-for-hypre.dp.cpp --match-full-lines %s
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/replace.h>
#include <thrust/remove.h>

struct greater_than_zero
{
  __host__ __device__
  bool operator()(int x) const
  {
    return x > 0;
  }
};

int main(){
    return 0;
}

void foo_host(){
  //CHECK: oneapi::dpl::equal_to<int>();
  thrust::equal_to<int>();
  //CHECK: oneapi::dpl::less<int>();
  thrust::less<int>();
}
