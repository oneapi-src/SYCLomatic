// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct -out-root %T/thrust_unsupported %s --cuda-include-path="%cuda-path/include" --usm-level=none
// RUN: FileCheck --input-file %T/thrust_unsupported/thrust_unsupported.dp.cpp --match-full-lines %s
#include <thrust/host_vector.h>
#include <thrust/replace.h>

namespace thrust {
   void replace_copy_if(int a, int b, int c) {
   }
};

int main() {
// CHECK: /*
// CHECK: DPCT1107:{{[0-9]+}}: Migration for this overload of thrust::replace_copy_if is not supported.
// CHECK: */
  thrust::replace_copy_if(1,2,3);
}