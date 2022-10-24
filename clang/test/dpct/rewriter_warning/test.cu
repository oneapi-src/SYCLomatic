// RUN: dpct --format-range=none -out-root %T/rewriter_warning %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/rewriter_warning/test.h
#include "test.h"

// CHECK: /*
// CHECK: DPCT1010:{{[0-9]+}}: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
// CHECK: */
int main()
{
    cudaCheckError("Memcopy device to device");
};