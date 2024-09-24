// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct -in-root %S -out-root %T/using_namespace %S/using_namespace.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/using_namespace/using_namespace.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl -DBUILD_TEST  %T/using_namespace/using_namespace.dp.cpp -o %T/using_namespace/using_namespace.dp.o %}

#include <cub/cub.cuh>
#include <cuda_runtime.h>

// CHECK: using namespace std;
// CHECK-EMPTY:
// CHECK-NEXT: void foo(sycl::plus<> sum) {}
using namespace std;
using namespace cub;

__global__ void foo(Sum sum) {}
int main() {
  Sum sum;
  foo<<<1, 1>>>(sum);
  return 0;
}
