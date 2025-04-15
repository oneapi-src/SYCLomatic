// REQUIRES: system-linux
// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1
// RUN: dpct --format-range=none -out-root %T/nvshmem %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/nvshmem/coll_ops.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST %T/nvshmem/coll_ops.dp.cpp -o %T/nvshmem/coll_ops.dp.o %}
#include <nvshmem.h>
#include <nvshmemx.h>

int main() {
  cudaStream_t stream = 0;

  // CHECK: ishmemx_barrier_all_on_queue(*stream);
  nvshmemx_barrier_all_on_stream(stream);

  return 0;
}
