// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4
// RUN: dpct --format-range=none -out-root %T/cublas_115 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublas_115/cublas_115.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/cublas_115/cublas_115.dp.cpp -o %T/cublas_115/cublas_115.dp.o %}

#include <cstdio>
#include <cublas_v2.h>

void foo1(cublasStatus_t s) {
  //CHECK:/*
  //CHECK-NEXT:DPCT1009:{{[0-9]+}}: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced by a placeholder string. You need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:printf("Error string: %s", "<Placeholder string>");
  printf("Error string: %s", cublasGetStatusString(s));
}
