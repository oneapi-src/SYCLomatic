// UNSUPPORTED: system-windows
// RUN: cd %T
// RUN: cat %S/compile_commands.json > %T/compile_commands.json
// RUN: cat %S/float_vector.cpp > %T/float_vector.cpp
// RUN: dpct -p %T -out-root %T/float_vector  --cuda-include-path="%cuda-path/include" 
// RUN: FileCheck %s --match-full-lines --input-file %T/float_vector/float_vector.cpp.dp.cpp


#include <cuda_runtime.h>
namespace quda {
  // CHECK: namespace dpct_operator_overloading {
  // CHECK-EMPTY:
  // CHECK-NEXT: inline sycl::double2 operator+(const sycl::double2 &x, const sycl::double2 &y)
  __host__ __device__ inline double2 operator+(const double2 &x, const double2 &y)
  {
  }
  // CHECK:} // namespace dpct_operator_overloading
}
int main() {
    return 0;
}
