// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct -out-root %T/cuda_atomic %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -fno-delayed-template-parsing
// RUN: FileCheck %s --match-full-lines --input-file %T/cuda_atomic/cuda_atomic.dp.cpp

#include <cuda/atomic>

int main(){
    // CHECK: atomic_ext<int, sycl::memory_order::relaxed, sycl::memory_scope::system> a;
    cuda::atomic<int, cuda::thread_scope_system> a;
}