// RUN: dpct --format-range=none -out-root %T/cpp_lang_extensions %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/cpp_lang_extensions/cpp_lang_extensions.dp.cpp --match-full-lines %s

__device__ float df(float f) {
  float a[23];
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to __ldg was removed, because there is no correspoinding API in DPC++.
  // CHECK-NEXT: */
  // CHECK-NEXT: f;
  __ldg(&f);
  int *pi;
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to __ldg was removed, because there is no correspoinding API in DPC++.
  // CHECK-NEXT: */
  // CHECK-NEXT: *pi;
  __ldg(pi);
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to __ldg was removed, because there is no correspoinding API in DPC++.
  // CHECK-NEXT: */
  // CHECK-NEXT: *(pi + 2);
  __ldg(pi + 2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to __ldg was removed, because there is no correspoinding API in DPC++.
  // CHECK-NEXT: */
  // CHECK-NEXT: return 45 * a[23] * f * 23;
  return 45 * __ldg(&a[23]) * f * 23;
}

