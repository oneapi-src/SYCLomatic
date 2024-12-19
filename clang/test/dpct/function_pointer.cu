// RUN: dpct --format-range=none -out-root %T/function_pointer %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/function_pointer/function_pointer.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/function_pointer/function_pointer.dp.cpp -o %T/function_pointer/function_pointer.dp.o %}

#include <cuda_runtime.h>
#include <iostream>

__global__ void vectorAdd(const int *A, int *B, int *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// CHECK:  void vectorAdd_wrapper(const int * A ,int * B ,int * C ,int N) {
// CHECK:        sycl::queue queue = *dpct::kernel_launcher::_que;
// CHECK:        unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
// CHECK:        sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;
// CHECK:        queue.parallel_for(
// CHECK:          nr,
// CHECK:          [=](sycl::nd_item<3> item_ct1) {
// CHECK:            vectorAdd(A, B, C, N, item_ct1);
// CHECK:          });
// CHECK:  }

template<typename T>
__global__ void vectorTemplateAdd(const T *A, T *B, T *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// CHECK:  template<typename T>
// CHECK:  void vectorTemplateAdd_wrapper(const T * A ,T * B ,T * C ,int N) {
// CHECK:      sycl::queue queue = *dpct::kernel_launcher::_que;
// CHECK:      unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
// CHECK:      sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;
// CHECK:      queue.parallel_for(
// CHECK:        nr,
// CHECK:        [=](sycl::nd_item<3> item_ct1) {
// CHECK:          vectorTemplateAdd<T>(A, B, C, N, item_ct1);
// CHECK:        });
// CHECK:  }

template <typename T>
using fpt = void(*)(const T *, T*, T*, int);

void foo() {
    int N = 10;
    size_t size = N * sizeof(int);

    int *h_A = new int[N];
    int *h_B = new int[N];
    int *h_C = new int[N];

    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<int>(i);
        h_B[i] = static_cast<int>(i * 2);
    }

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

// CHECK:  fpt<int> fp = dpct::wrapper_register(vectorAdd_wrapper).get();
// CHECK:  dpct::kernel_launcher::launch(fp, 1, 10, 0, 0, d_A, d_B, d_C, N);
    fpt<int> fp = vectorAdd;
    fp<<<1, 10>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "Result: " << std::endl;
    for (int i = 0; i < N; ++i) {
        if(h_A[i] + h_B[i] != h_C[i]) {
            std::cout << "test failed" << std::endl;
            exit(-1);
        }
        std::cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << std::endl;
    }
    
    void *args[4];
    args[0] = &d_A;
    args[1] = &d_B;
    args[2] = &d_C;
    args[3] = &N;
    // CHECK:  dpct::kernel_launcher::launch(fp, 1, 10, args, 0, 0);
    cudaLaunchKernel((void *)fp, 1, 10, args, 0, 0);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "Result: " << std::endl;
    for (int i = 0; i < N; ++i) {
        if(h_A[i] + h_B[i] != h_C[i]) {
            std::cout << "test failed" << std::endl;
            exit(-1);
        }
        std::cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << std::endl;
    }

    // CHECK:  dpct::kernel_launcher::launch(fp, 1, 10, args, 0, 0);
    cudaLaunchKernel<void(const int*, int*, int*, int)>(fp, 1, 10, args, 0, 0);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "Result: " << std::endl;
    for (int i = 0; i < N; ++i) {
        if(h_A[i] + h_B[i] != h_C[i]) {
            std::cout << "test failed" << std::endl;
            exit(-1);
        }
        std::cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << std::endl;
    }


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}

template<typename T>
void goo(fpt<T> p) {
    int N = 10;
    size_t size = N * sizeof(int);

    int *h_A = new int[N];
    int *h_B = new int[N];
    int *h_C = new int[N];

    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<int>(i);
        h_B[i] = static_cast<int>(i * 2);
    }

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    // CHECK:  dpct::kernel_launcher::launch(p, 1, 10, 0, 0, d_A, d_B, d_C, N);
    p<<<1, 10>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "Result: " << std::endl;
    for (int i = 0; i < N; ++i) {
        if(h_A[i] + h_B[i] != h_C[i]) {
            std::cout << "test failed" << std::endl;
            exit(-1);
        }
        std::cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}

template <typename T>
void hoo() {
    // CHECK: fpt<int> a = dpct::wrapper_register<decltype(a)>(vectorTemplateAdd_wrapper).get();
  fpt<int> a = vectorTemplateAdd;
  goo<int>(a);
  // CHECK:  goo<T>(dpct::wrapper_register<typename dpct::nth_argument_type<decltype(goo<T>), 0>::type>(vectorTemplateAdd_wrapper).get());
  goo<T>(vectorTemplateAdd);
}

int main() {
  hoo<int>();
  foo();
  std::cout << "test success" << std::endl;
  return 0;
}
