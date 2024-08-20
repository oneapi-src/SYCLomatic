// UNSUPPORTED: v7.0, v7.5, v8.0, v9.0, v9.2, v10.0, v10.1, v10.2
// UNSUPPORTED: cuda-7.0, cuda-7.5, cuda-8.0, cuda-9.0, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/Libcu %S/libcu_atomic_using_ns.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/Libcu/libcu_atomic_using_ns.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/Libcu/libcu_atomic_using_ns.dp.cpp -o %T/Libcu/libcu_atomic_using_ns.dp.o %}

// CHECK: #include <sycl/sycl.hpp>
// CHECK: #include <dpct/dpct.hpp>
// CHECK: #include <dpct/atomic.hpp>
#include <cuda/atomic>

// CHECK: // BEGIN
// CHECK-EMPTY:
// CHECK-NEXT: // END
// BEGIN
using namespace cuda;
// END

int main() {
  // CHECK: sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
  atomic_thread_fence(cuda::std::memory_order_release, thread_scope_system);

  return 0;
}
