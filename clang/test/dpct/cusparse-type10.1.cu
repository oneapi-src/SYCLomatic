// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1
// RUN: dpct --format-range=none --out-root %T/cusparse-type10 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cusparse-type10/cusparse-type10.1.dp.cpp --match-full-lines %s
#include <cstdio>
#include <cusparse_v2.h>
#include <cuda_runtime.h>

// CUSPARSE_STATUS_NOT_SUPPORTED is available since v10.2.
int main(){
  //CHECK: int a6;
  //CHECK-NEXT: a6 = 10;
  cusparseStatus_t a6;
  a6 = CUSPARSE_STATUS_NOT_SUPPORTED;
}

