// UNSUPPORTED: v7.0, v7.5, v8.0, v9.0, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1
// UNSUPPORTED: cuda-7.0, cuda-7.5, cuda-8.0, cuda-9.0, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1
// RUN: dpct --format-range=none -in-root %S -out-root %T/Libcu %S/libcu_std_atomic.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/Libcu/libcu_std_atomic.dp.cpp --match-full-lines %s


// CHECK: #include <sycl/sycl.hpp>
// CHECK: #include <dpct/dpct.hpp>
// CHECK: #include <dpct/atomic.hpp>
#include <cuda/std/atomic>


template <typename T>
void device_construct() {}

int main(){
  // CHECK: sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
  cuda::std::atomic_thread_fence(cuda::std::memory_order_release);
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
  ans = a.fetch_add(1);
  ans = a.fetch_sub(-1);

  //CHECK: device_construct<dpct::atomic<signed int>>();
  device_construct<cuda::std::atomic<signed int>>();

  return 0;
}