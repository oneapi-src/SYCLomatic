// RUN: dpct -out-root %T/cuda_atomic %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -fno-delayed-template-parsing
// RUN: FileCheck %s --match-full-lines --input-file %T/cuda_atomic/cuda_atomic.dp.cpp

#include <cuda/atomic>

int main(){
    // CHECK: cuda::atomic_ext<int, sycl::memory_order::relaxed, sycl::memory_scope::system> a;
    cuda::atomic<int, cuda::thread_scope_system> a;
}