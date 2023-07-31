// RUN: dpct --format-range=none --usm-level=none -out-root %T/atomic_functions_system_wide %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/atomic_functions_system_wide/atomic_functions_system_wide.dp.cpp --match-full-lines %s

#include <cuda_runtime.h>

#include <iostream>
#include <memory>

// CHECK:void atomic_kernel(int *atomic_array, const sycl::nd_item<3> &item_ct1) {
// CHECK-NEXT:  unsigned int tid = item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);
// CHECK-NEXT:  dpct::atomic_fetch_add<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(&atomic_array[0], 10);
// CHECK-NEXT:  dpct::atomic_exchange<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(&atomic_array[1], tid);
// CHECK-NEXT:  dpct::atomic_fetch_max<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(&atomic_array[2], tid);
// CHECK-NEXT:  dpct::atomic_fetch_min<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(&atomic_array[3], tid);
// CHECK-NEXT:  dpct::atomic_fetch_compare_inc<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>((unsigned int *)&atomic_array[4], 17);
// CHECK-NEXT:  dpct::atomic_fetch_compare_dec<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>((unsigned int *)&atomic_array[5], 137);
// CHECK-NEXT:  dpct::atomic_compare_exchange_strong<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(&atomic_array[6], tid - 1, tid);
// CHECK-NEXT:  dpct::atomic_fetch_and<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(&atomic_array[7], 2 * tid + 7);
// CHECK-NEXT:  dpct::atomic_fetch_or<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(&atomic_array[8], 1 << tid);
// CHECK-NEXT:  dpct::atomic_fetch_xor<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(&atomic_array[9], tid);
// CHECK-NEXT:}
__global__ void atomic_kernel(int *atomic_array) {
  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
  atomicAdd_system(&atomic_array[0], 10);
  atomicExch_system(&atomic_array[1], tid);
  atomicMax_system(&atomic_array[2], tid);
  atomicMin_system(&atomic_array[3], tid);
  atomicInc_system((unsigned int *)&atomic_array[4], 17);
  atomicDec_system((unsigned int *)&atomic_array[5], 137);
  atomicCAS_system(&atomic_array[6], tid - 1, tid);
  atomicAnd_system(&atomic_array[7], 2 * tid + 7);
  atomicOr_system(&atomic_array[8], 1 << tid);
  atomicXor_system(&atomic_array[9], tid);
}
