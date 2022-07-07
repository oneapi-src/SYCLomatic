// UNSUPPORTED: v7.0, v7.5, v8.0, v9.0, v9.2, v10.0, v10.1, v10.2
// UNSUPPORTED: cuda-7.0, cuda-7.5, cuda-8.0, cuda-9.0, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// RUN: dpct -in-root %S -out-root %T/Libcu %S/libcu_std_atomic.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/Libcu/libcu_std_atomic.dp.cpp --match-full-lines %s


// CHECK: #include <CL/sycl.hpp>
// CHECK: #include <dpct/dpct.hpp>
// CHECK: #include <dpct/atomic_utils.hpp>
#include <cuda/std/atomic>

int main(){
  // CHECK: dpct::atomic<int> a;
  // CHECK: dpct::atomic<int> b(0);
  // CHECK: dpct::atomic<int> c(0);
  // CHECK: int ans = c.load();
  // CHECK: a.store(0);
  // CHECK: int ans2 = a.load();
  cuda::atomic<int> a;
  cuda::atomic<int> b(0);
  cuda::atomic<int> c(0);
  int ans = c.load();
  a.store(0);
  int ans2 = a.load();

  //CHECK: int tmp =1,tmp1=2;
  //CHECK: ans = a.exchange(1);
  //CHECK: a.compare_exchange_weak(tmp,2);
  //CHECK: a.compare_exchange_strong(tmp1,3);
  int tmp =1,tmp1=2;
  ans = a.exchange(1);
  a.compare_exchange_weak(tmp,2);
  a.compare_exchange_strong(tmp1,3);

  //CHECK: ans = a.fetch_add(1);
  //CHECK: ans = a.fetch_sub(-1);
  //CHECK: ans = a.fetch_and(2);
  //CHECK: ans = a.fetch_or(1);
  //CHECK: ans = a.fetch_xor(2);
  ans = a.fetch_add(1);
  ans = a.fetch_sub(-1);
  ans = a.fetch_and(2);
  ans = a.fetch_or(1);
  ans = a.fetch_xor(2);

  return 0;
}
