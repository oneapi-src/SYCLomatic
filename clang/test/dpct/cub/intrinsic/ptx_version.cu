// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/intrinsic/ptx_version %S/ptx_version.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/intrinsic/ptx_version/ptx_version.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/intrinsic/ptx_version/ptx_version.dp.cpp -o %T/intrinsic/ptx_version/ptx_version.dp.o %}

#include <cub/cub.cuh>

void test() {
  int a = 0;
  // CHECK: a = DPCT_COMPATIBILITY_TEMP;
  // CHECK-NEXT: a = DPCT_COMPATIBILITY_TEMP;
  // CHECK-NEXT: a = DPCT_COMPATIBILITY_TEMP;
  // CHECK-NEXT: a = DPCT_COMPATIBILITY_TEMP;
  cub::PtxVersion(a);
  cub::PtxVersion(a, 0);
  cub::PtxVersionUncached(a);
  cub::PtxVersionUncached(a, 0);
}
