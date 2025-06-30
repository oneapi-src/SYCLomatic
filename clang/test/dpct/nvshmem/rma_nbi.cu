// REQUIRES: system-linux
// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7
// RUN: dpct --format-range=none -out-root %T/nvshmem %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/nvshmem/rma_nbi.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST %T/nvshmem/rma_nbi.dp.cpp -o %T/nvshmem/rma_nbi.dp.o %}
#include <nvshmem.h>
#include <nvshmemx.h>


__host__ __device__ void test(int target_pe) {
  const void *src_void;
  void *dst_void;

  const int count = 10;

  // nvshmem_putmem_nbi
  // ishmem_putmem_nbi
  // CHECK: ishmem_putmem_nbi(dst_void, src_void, count, target_pe);
  nvshmem_putmem_nbi(dst_void, src_void, count, target_pe);
}

int main() {
  const void *src_void;
  void *dst_void;

  int target_pe = 0;
  const int count = 10;

  // nvshmem_putmem_nbi
  // ishmem_putmem_nbi
  // CHECK: ishmem_putmem_nbi(dst_void, src_void, count, target_pe);
  nvshmem_putmem_nbi(dst_void, src_void, count, target_pe);
}
