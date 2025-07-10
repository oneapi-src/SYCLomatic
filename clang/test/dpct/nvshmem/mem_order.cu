// REQUIRES: system-linux
// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1
// RUN: dpct --format-range=none -out-root %T/nvshmem %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/nvshmem/mem_order.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST %T/nvshmem/mem_order.dp.cpp -o %T/nvshmem/mem_order.dp.o %}
#include <nvshmem.h>
#include <nvshmemx.h>

__host__ __device__ void test() {
  // CHECK: ishmem_fence();
  nvshmem_fence();

  // CHECK: ishmem_quiet();
  nvshmem_quiet();
}

int main() {
  cudaStream_t stream;

  // CHECK: ishmem_fence();
  nvshmem_fence();

  // CHECK: ishmem_quiet();
  nvshmem_quiet();

  // ishmemx_quiet_on_queue(*stream);
  nvshmemx_quiet_on_stream(stream);

  return 0;
}
