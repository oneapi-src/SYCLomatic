// UNSUPPORTED: system-windows
// RUN: dpct --format-range=none -out-root %T/allocate_memory_kernel %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/allocate_memory_kernel/allocate_memory_kernel.dp.cpp
#include <cstdlib>
#include <cuda.h>
#include <stdio.h>

#include <cuda_runtime.h>
template <typename T>
class TestVirtualFunc {
public:
  __device__ TestVirtualFunc() {}
  // CHECK: /*
  // CHECK-NEXT: DPCT1109:{{[0-9]+}}:  Virtual functions cannot be called in SYCL device code. You need to adjust the code.
  // CHECK-NEXT: */
  __device__ virtual ~TestVirtualFunc() {}
  // CHECK: /*
  // CHECK-NEXT: DPCT1109:{{[0-9]+}}:  Virtual functions cannot be called in SYCL device code. You need to adjust the code.
  // CHECK-NEXT: */
  __device__ virtual void push(const T &&e) = 0;
};
template <typename T>
class TestSeqContainer : public TestVirtualFunc<T> {
public:
  // CHECK: /*
  // CHECK-NEXT: DPCT1109:{{[0-9]+}}: The usage of dynamic memory allocation and deallocation APIs cannot be called in SYCL device code. You need to adjust the code.
  // CHECK-NEXT: */
  __device__ TestSeqContainer(int size) : index_top(-1) { m_data = new T[size]; }

  __device__ ~TestSeqContainer() {
    if (m_data)
  // CHECK: /*
  // CHECK-NEXT: DPCT1109:{{[0-9]+}}: The usage of dynamic memory allocation and deallocation APIs cannot be called in SYCL device code. You need to adjust the code.
  // CHECK-NEXT: */
      delete[] m_data;
  }
  // CHECK: /*
  // CHECK-NEXT: DPCT1109:{{[0-9]+}}:  Virtual functions cannot be called in SYCL device code. You need to adjust the code.
  // CHECK-NEXT: */
  __device__ virtual void push(const T &&e) {
    if (m_data) {
      int idx = atomicAdd(&this->index_top, 1);
      m_data[idx] = e;
    }
  }

private:
  T *m_data;
  int index_top;
};
__global__ void func() {
  // CHECK: /*
  // CHECK-NEXT: DPCT1109:{{[0-9]+}}: The usage of dynamic memory allocation and deallocation APIs cannot be called in SYCL device code. You need to adjust the code.
  // CHECK-NEXT: */
  auto seq = new TestSeqContainer<int>(10);
  // CHECK: /*
  // CHECK-NEXT: DPCT1109:{{[0-9]+}}:  Virtual functions cannot be called in SYCL device code. You need to adjust the code.
  // CHECK-NEXT: */
  seq->push(10);
  // CHECK: /*
  // CHECK-NEXT: DPCT1109:{{[0-9]+}}: The usage of dynamic memory allocation and deallocation APIs cannot be called in SYCL device code. You need to adjust the code.
  // CHECK-NEXT: */
  int *test = (int *)malloc(10 * sizeof(10));
  // CHECK: /*
  // CHECK-NEXT: DPCT1109:{{[0-9]+}}: The usage of dynamic memory allocation and deallocation APIs cannot be called in SYCL device code. You need to adjust the code.
  // CHECK-NEXT: */
  free(test);
  // CHECK: /*
  // CHECK-NEXT: DPCT1109:{{[0-9]+}}: The usage of dynamic memory allocation and deallocation APIs cannot be called in SYCL device code. You need to adjust the code.
  // CHECK-NEXT: */
  delete seq;
  // CHECK: /*
  // CHECK-NEXT: DPCT1109:{{[0-9]+}}: The usage of dynamic memory allocation and deallocation APIs cannot be called in SYCL device code. You need to adjust the code.
  // CHECK-NEXT: */
  int4 *ptr = (int4 *)alloca(10 * sizeof(int4));
}

int main() {
  func<<<1, 1>>>();
  cudaDeviceSynchronize();
  return 0;
}
