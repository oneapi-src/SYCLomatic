// REQUIRES: system-linux
// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7
// RUN: dpct --format-range=none -out-root %T/nvshmem %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/nvshmem/mem_mgmt.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST %T/nvshmem/mem_mgmt.dp.cpp -o %T/nvshmem/mem_mgmt.dp.o %}
#include <nvshmem.h>
#include <nvshmemx.h>

int main() {
  int my_pe = 0;
  size_t size = 1024;

  // CHECK: int* array = (int*)ishmem_malloc(size * sizeof(int));
  // CHECK-NEXT: array = (int*)ishmem_align(64, size * sizeof(int));
  // CHECK-NEXT: array = (int*)ishmem_calloc(size, sizeof(int));
  // CHECK-NEXT: void* symmetric_ptr = ishmem_ptr(array, my_pe);
  int* array = (int*)nvshmem_malloc(size * sizeof(int));
  array = (int*)nvshmem_align(64, size * sizeof(int));
  array = (int*)nvshmem_calloc(size, sizeof(int));
  void* symmetric_ptr = nvshmem_ptr(array, my_pe);

  // CHECK: ishmem_free(array);
  nvshmem_free(array);
}
