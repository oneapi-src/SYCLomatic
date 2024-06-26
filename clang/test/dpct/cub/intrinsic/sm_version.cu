// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/intrinsic/sm_version %S/sm_version.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/intrinsic/sm_version/sm_version.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/intrinsic/sm_version/sm_version.dp.cpp -o %T/intrinsic/sm_version/sm_version.dp.o %}

#include <cub/cub.cuh>

void test() {
  int a = 0;
  // CHECK: a = dpct::get_major_version(dev_ct1) * 100 + dpct::get_minor_version(dev_ct1) * 10;
  // CHECK-NEXT: a = dpct::get_major_version(dev_ct1) * 100 + dpct::get_minor_version(dev_ct1) * 10;
  // CHECK-NEXT: a = dpct::get_major_version(dev_ct1) * 100 + dpct::get_minor_version(dev_ct1) * 10;
  // CHECK-NEXT: a = dpct::get_major_version(dev_ct1) * 100 + dpct::get_minor_version(dev_ct1) * 10;
  cub::SmVersion(a);
  cub::SmVersion(a, 0);
  cub::SmVersionUncached(a);
  cub::SmVersionUncached(a, 0);
}
