// RUN: dpct --format-range=none -out-root %T/function_pointer2 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/function_pointer2/function_pointer2.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/function_pointer2/function_pointer2.dp.cpp -o %T/function_pointer2/function_pointer2.dp.o %}

#include <cuda_runtime.h>
#include <iostream>

template<typename T>
__global__ static inline void vectorTemplateAdd(const T *A, T *B, T *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

template <typename T>
using fpt = void(*)(const T *, T*, T*, int);

template<typename T>
void foo() {
    int *d_A, *d_B, *d_C;
    // CHECK:  fpt<T> fp = &vectorTemplateAdd_wrapper<T>;
    fpt<T> fp = &vectorTemplateAdd<T>;
    // CHECK:  dpct::kernel_launcher::launch(fp, 1, 10, 0, 0, d_A, d_B, d_C, 10);
    fp<<<1, 10>>>(d_A, d_B, d_C, 10);
}

static __global__ void setup_kernel(int p){}

template<typename T>
void goo();

template<typename T>
void goo() {
  // CHECK: auto a = (void *)setup_kernel_wrapper;
  auto a = (void *)setup_kernel;
}

template void goo<int>();

int main() {
  foo<int>();
  std::cout << "test success" << std::endl;
  return 0;
}
