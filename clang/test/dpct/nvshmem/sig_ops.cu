// REQUIRES: system-linux
// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.2
// RUN: dpct --format-range=none -out-root %T/nvshmem %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/nvshmem/sig_ops.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST %T/nvshmem/sig_ops.dp.cpp -o %T/nvshmem/sig_ops.dp.o %}
// CHECK: #include <ishmem.h>
// CHECK-NEXT: #include <ishmemx.h>
// CHECK-NEXT: #include <dpct/shmem_utils.hpp>
#include <nvshmem.h>
#include <nvshmemx.h>


__host__ __device__ void test() {
  void *src_void;
  void *dst_void;
  int nelems;

  int pe;
  uint64_t sig_addr, signal;

  // CHECK: int sig_op = ISHMEM_SIGNAL_SET;
  // CHECK-NEXT: sig_op = ISHMEM_SIGNAL_ADD;
  nvshmemi_amo_t sig_op = NVSHMEM_SIGNAL_SET;
  sig_op = NVSHMEM_SIGNAL_ADD;

  // CHECK: int comp = ISHMEM_CMP_EQ;
  // CHECK-NEXT: comp = ISHMEM_CMP_NE;
  // CHECK-NEXT: comp = ISHMEM_CMP_GT;
  // CHECK-NEXT: comp = ISHMEM_CMP_GE;
  // CHECK-NEXT: comp = ISHMEM_CMP_LT;
  // CHECK-NEXT: comp = ISHMEM_CMP_LE;
  nvshmemi_cmp_type comp = NVSHMEM_CMP_EQ;
  comp = NVSHMEM_CMP_NE;
  comp = NVSHMEM_CMP_GT;
  comp = NVSHMEM_CMP_GE;
  comp = NVSHMEM_CMP_LT;
  comp = NVSHMEM_CMP_LE;

  // nvshmem_signal_wait_until
  // ishmem_signal_wait_until
  // CHECK: ishmem_signal_wait_until(&sig_addr, comp, signal);
  nvshmem_signal_wait_until(&sig_addr, comp, signal);

  // nvshmem_putmem_signal_nbi
  // ishmem_putmem_signal_nbi
  // CHECK: ishmem_putmem_signal_nbi(dst_void, src_void, nelems, &sig_addr, signal, sig_op, pe);
  nvshmem_putmem_signal_nbi(dst_void, src_void, nelems, &sig_addr, signal, sig_op, pe);

  // nvshmemx_signal_op
  // ishmemx_signal_set/add
  // CHECK: dpct::shmemx::signal_op(&sig_addr, signal, sig_op, pe);
  // CHECK-NEXT: dpct::shmemx::signal_op(&sig_addr, signal, ISHMEM_SIGNAL_SET, pe);
  // CHECK-NEXT: dpct::shmemx::signal_op(&sig_addr, signal, ISHMEM_SIGNAL_ADD, pe);
  nvshmemx_signal_op(&sig_addr, signal, sig_op, pe);
  nvshmemx_signal_op(&sig_addr, signal, NVSHMEM_SIGNAL_SET, pe);
  nvshmemx_signal_op(&sig_addr, signal, NVSHMEM_SIGNAL_ADD, pe);
}
