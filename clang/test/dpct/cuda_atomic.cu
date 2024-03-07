// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct -out-root %T/cuda_atomic %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/cuda_atomic/cuda_atomic.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/cuda_atomic/cuda_atomic.dp.cpp -o %T/cuda_atomic/cuda_atomic.dp.o %}

#include <cuda/atomic>

int main(){
    // CHECK: dpct::atomic<int, sycl::memory_scope::system, sycl::memory_order::relaxed> a;
    cuda::atomic<int, cuda::thread_scope_system> a;
}