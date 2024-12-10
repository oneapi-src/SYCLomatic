// RUN: dpct --format-range=none -out-root %T/cpp_lang_extensions %s --cuda-include-path="%cuda-path/include" -extra-arg="-I%S" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/cpp_lang_extensions/cpp_lang_extensions.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/cpp_lang_extensions/cpp_lang_extensions.dp.cpp -o %T/cpp_lang_extensions/cpp_lang_extensions.dp.o %}

#include "cpp_lang_extensions.cuh"

__device__ float df(float f) {
  float a[23];
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: f;
  __ldg(&f);
  int *pi;
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *pi;
  __ldg(pi);
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *(pi + 2);
  __ldg(pi + 2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: return 45 * a[23] * f * 23;
  return 45 * __ldg(&a[23]) * f * 23;
}

// CHECK: /*
// CHECK-NEXT: DPCT1110:{{[0-9]+}}: The total declared local variable size in device function dev exceeds 128 bytes and may cause high register pressure. Consult with your hardware vendor to find the total register size available and adjust the code, or use smaller sub-group size to avoid high register pressure.
// CHECK-NEXT: */
__device__ void dev() {
  char *c_1;
  char c_2;
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: c_2 = *c_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldca call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: c_2 = *c_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: c_2 = *c_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldlu call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: c_2 = *c_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcv call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: c_2 = *c_1;
  c_2 = __ldcg(c_1);
  c_2 = __ldca(c_1);
  c_2 = __ldcs(c_1);
  c_2 = __ldlu(c_1);
  c_2 = __ldcv(c_1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwb call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *c_1 = c_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *c_1 = c_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *c_1 = c_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwt call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *c_1 = c_2;
  __stwb(c_1, c_2);
  __stcg(c_1, c_2);
  __stcs(c_1, c_2);
  __stwt(c_1, c_2);

  char2 *c2_1;
  char2 c2_2;
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: c2_2 = *c2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldca call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: c2_2 = *c2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: c2_2 = *c2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldlu call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: c2_2 = *c2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcv call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: c2_2 = *c2_1;
  c2_2 = __ldcg(c2_1);
  c2_2 = __ldca(c2_1);
  c2_2 = __ldcs(c2_1);
  c2_2 = __ldlu(c2_1);
  c2_2 = __ldcv(c2_1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwb call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *c2_1 = c2_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *c2_1 = c2_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *c2_1 = c2_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwt call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *c2_1 = c2_2;
  __stwb(c2_1, c2_2);
  __stcg(c2_1, c2_2);
  __stcs(c2_1, c2_2);
  __stwt(c2_1, c2_2);

  char4 *c4_1;
  char4 c4_2;
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: c4_2 = *c4_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldca call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: c4_2 = *c4_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: c4_2 = *c4_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldlu call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: c4_2 = *c4_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcv call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: c4_2 = *c4_1;
  c4_2 = __ldcg(c4_1);
  c4_2 = __ldca(c4_1);
  c4_2 = __ldcs(c4_1);
  c4_2 = __ldlu(c4_1);
  c4_2 = __ldcv(c4_1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwb call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *c4_1 = c4_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *c4_1 = c4_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *c4_1 = c4_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwt call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *c4_1 = c4_2;
  __stwb(c4_1, c4_2);
  __stcg(c4_1, c4_2);
  __stcs(c4_1, c4_2);
  __stwt(c4_1, c4_2);

  double *d_1;
  double d_2;
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: d_2 = *d_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldca call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: d_2 = *d_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: d_2 = *d_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldlu call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: d_2 = *d_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcv call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: d_2 = *d_1;
  d_2 = __ldcg(d_1);
  d_2 = __ldca(d_1);
  d_2 = __ldcs(d_1);
  d_2 = __ldlu(d_1);
  d_2 = __ldcv(d_1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwb call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *d_1 = d_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *d_1 = d_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *d_1 = d_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwt call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *d_1 = d_2;
  __stwb(d_1, d_2);
  __stcg(d_1, d_2);
  __stcs(d_1, d_2);
  __stwt(d_1, d_2);

  double2 *d2_1;
  double2 d2_2;
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2_2 = *d2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldca call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2_2 = *d2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2_2 = *d2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldlu call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2_2 = *d2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcv call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2_2 = *d2_1;
  d2_2 = __ldcg(d2_1);
  d2_2 = __ldca(d2_1);
  d2_2 = __ldcs(d2_1);
  d2_2 = __ldlu(d2_1);
  d2_2 = __ldcv(d2_1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwb call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *d2_1 = d2_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *d2_1 = d2_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *d2_1 = d2_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwt call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *d2_1 = d2_2;
  __stwb(d2_1, d2_2);
  __stcg(d2_1, d2_2);
  __stcs(d2_1, d2_2);
  __stwt(d2_1, d2_2);

  float *f_1;
  float f_2;
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: f_2 = *f_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldca call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: f_2 = *f_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: f_2 = *f_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldlu call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: f_2 = *f_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcv call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: f_2 = *f_1;
  f_2 = __ldcg(f_1);
  f_2 = __ldca(f_1);
  f_2 = __ldcs(f_1);
  f_2 = __ldlu(f_1);
  f_2 = __ldcv(f_1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwb call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *f_1 = f_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *f_1 = f_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *f_1 = f_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwt call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *f_1 = f_2;
  __stwb(f_1, f_2);
  __stcg(f_1, f_2);
  __stcs(f_1, f_2);
  __stwt(f_1, f_2);

  float2 *f2_1;
  float2 f2_2;
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2_2 = *f2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldca call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2_2 = *f2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2_2 = *f2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldlu call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2_2 = *f2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcv call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2_2 = *f2_1;
  f2_2 = __ldcg(f2_1);
  f2_2 = __ldca(f2_1);
  f2_2 = __ldcs(f2_1);
  f2_2 = __ldlu(f2_1);
  f2_2 = __ldcv(f2_1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwb call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *f2_1 = f2_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *f2_1 = f2_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *f2_1 = f2_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwt call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *f2_1 = f2_2;
  __stwb(f2_1, f2_2);
  __stcg(f2_1, f2_2);
  __stcs(f2_1, f2_2);
  __stwt(f2_1, f2_2);

  float4 *f4_1;
  float4 f4_2;
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: f4_2 = *f4_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldca call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: f4_2 = *f4_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: f4_2 = *f4_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldlu call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: f4_2 = *f4_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcv call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: f4_2 = *f4_1;
  f4_2 = __ldcg(f4_1);
  f4_2 = __ldca(f4_1);
  f4_2 = __ldcs(f4_1);
  f4_2 = __ldlu(f4_1);
  f4_2 = __ldcv(f4_1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwb call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *f4_1 = f4_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *f4_1 = f4_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *f4_1 = f4_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwt call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *f4_1 = f4_2;
  __stwb(f4_1, f4_2);
  __stcg(f4_1, f4_2);
  __stcs(f4_1, f4_2);
  __stwt(f4_1, f4_2);

  int *i_1;
  int i_2;
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: i_2 = *i_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldca call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: i_2 = *i_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: i_2 = *i_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldlu call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: i_2 = *i_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcv call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: i_2 = *i_1;
  i_2 = __ldcg(i_1);
  i_2 = __ldca(i_1);
  i_2 = __ldcs(i_1);
  i_2 = __ldlu(i_1);
  i_2 = __ldcv(i_1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwb call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *i_1 = i_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *i_1 = i_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *i_1 = i_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwt call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *i_1 = i_2;
  __stwb(i_1, i_2);
  __stcg(i_1, i_2);
  __stcs(i_1, i_2);
  __stwt(i_1, i_2);

  int2 *i2_1;
  int2 i2_2;
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: i2_2 = *i2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldca call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: i2_2 = *i2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: i2_2 = *i2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldlu call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: i2_2 = *i2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcv call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: i2_2 = *i2_1;
  i2_2 = __ldcg(i2_1);
  i2_2 = __ldca(i2_1);
  i2_2 = __ldcs(i2_1);
  i2_2 = __ldlu(i2_1);
  i2_2 = __ldcv(i2_1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwb call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *i2_1 = i2_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *i2_1 = i2_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *i2_1 = i2_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwt call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *i2_1 = i2_2;
  __stwb(i2_1, i2_2);
  __stcg(i2_1, i2_2);
  __stcs(i2_1, i2_2);
  __stwt(i2_1, i2_2);

  int4 *i4_1;
  int4 i4_2;
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: i4_2 = *i4_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldca call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: i4_2 = *i4_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: i4_2 = *i4_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldlu call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: i4_2 = *i4_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcv call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: i4_2 = *i4_1;
  i4_2 = __ldcg(i4_1);
  i4_2 = __ldca(i4_1);
  i4_2 = __ldcs(i4_1);
  i4_2 = __ldlu(i4_1);
  i4_2 = __ldcv(i4_1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwb call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *i4_1 = i4_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *i4_1 = i4_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *i4_1 = i4_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwt call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *i4_1 = i4_2;
  __stwb(i4_1, i4_2);
  __stcg(i4_1, i4_2);
  __stcs(i4_1, i4_2);
  __stwt(i4_1, i4_2);

  long *l_1;
  long l_2;
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: l_2 = *l_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldca call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: l_2 = *l_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: l_2 = *l_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldlu call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: l_2 = *l_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcv call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: l_2 = *l_1;
  l_2 = __ldcg(l_1);
  l_2 = __ldca(l_1);
  l_2 = __ldcs(l_1);
  l_2 = __ldlu(l_1);
  l_2 = __ldcv(l_1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwb call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *l_1 = l_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *l_1 = l_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *l_1 = l_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwt call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *l_1 = l_2;
  __stwb(l_1, l_2);
  __stcg(l_1, l_2);
  __stcs(l_1, l_2);
  __stwt(l_1, l_2);

  long long *ll_1;
  long long ll_2;
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ll_2 = *ll_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldca call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ll_2 = *ll_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ll_2 = *ll_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldlu call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ll_2 = *ll_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcv call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ll_2 = *ll_1;
  ll_2 = __ldcg(ll_1);
  ll_2 = __ldca(ll_1);
  ll_2 = __ldcs(ll_1);
  ll_2 = __ldlu(ll_1);
  ll_2 = __ldcv(ll_1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwb call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *ll_1 = ll_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *ll_1 = ll_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *ll_1 = ll_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwt call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *ll_1 = ll_2;
  __stwb(ll_1, ll_2);
  __stcg(ll_1, ll_2);
  __stcs(ll_1, ll_2);
  __stwt(ll_1, ll_2);

  longlong2 *ll2_1;
  longlong2 ll2_2;
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ll2_2 = *ll2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldca call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ll2_2 = *ll2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ll2_2 = *ll2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldlu call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ll2_2 = *ll2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcv call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ll2_2 = *ll2_1;
  ll2_2 = __ldcg(ll2_1);
  ll2_2 = __ldca(ll2_1);
  ll2_2 = __ldcs(ll2_1);
  ll2_2 = __ldlu(ll2_1);
  ll2_2 = __ldcv(ll2_1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwb call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *ll2_1 = ll2_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *ll2_1 = ll2_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *ll2_1 = ll2_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwt call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *ll2_1 = ll2_2;
  __stwb(ll2_1, ll2_2);
  __stcg(ll2_1, ll2_2);
  __stcs(ll2_1, ll2_2);
  __stwt(ll2_1, ll2_2);

  short *s_1;
  short s_2;
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: s_2 = *s_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldca call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: s_2 = *s_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: s_2 = *s_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldlu call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: s_2 = *s_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcv call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: s_2 = *s_1;
  s_2 = __ldcg(s_1);
  s_2 = __ldca(s_1);
  s_2 = __ldcs(s_1);
  s_2 = __ldlu(s_1);
  s_2 = __ldcv(s_1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwb call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *s_1 = s_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *s_1 = s_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *s_1 = s_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwt call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *s_1 = s_2;
  __stwb(s_1, s_2);
  __stcg(s_1, s_2);
  __stcs(s_1, s_2);
  __stwt(s_1, s_2);

  short2 *s2_1;
  short2 s2_2;
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: s2_2 = *s2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldca call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: s2_2 = *s2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: s2_2 = *s2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldlu call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: s2_2 = *s2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcv call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: s2_2 = *s2_1;
  s2_2 = __ldcg(s2_1);
  s2_2 = __ldca(s2_1);
  s2_2 = __ldcs(s2_1);
  s2_2 = __ldlu(s2_1);
  s2_2 = __ldcv(s2_1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwb call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *s2_1 = s2_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *s2_1 = s2_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *s2_1 = s2_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwt call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *s2_1 = s2_2;
  __stwb(s2_1, s2_2);
  __stcg(s2_1, s2_2);
  __stcs(s2_1, s2_2);
  __stwt(s2_1, s2_2);

  short4 *s4_1;
  short4 s4_2;
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: s4_2 = *s4_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldca call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: s4_2 = *s4_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: s4_2 = *s4_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldlu call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: s4_2 = *s4_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcv call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: s4_2 = *s4_1;
  s4_2 = __ldcg(s4_1);
  s4_2 = __ldca(s4_1);
  s4_2 = __ldcs(s4_1);
  s4_2 = __ldlu(s4_1);
  s4_2 = __ldcv(s4_1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwb call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *s4_1 = s4_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *s4_1 = s4_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *s4_1 = s4_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwt call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *s4_1 = s4_2;
  __stwb(s4_1, s4_2);
  __stcg(s4_1, s4_2);
  __stcs(s4_1, s4_2);
  __stwt(s4_1, s4_2);

  signed char *sc_1;
  signed char sc_2;
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: sc_2 = *sc_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldca call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: sc_2 = *sc_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: sc_2 = *sc_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldlu call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: sc_2 = *sc_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcv call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: sc_2 = *sc_1;
  sc_2 = __ldcg(sc_1);
  sc_2 = __ldca(sc_1);
  sc_2 = __ldcs(sc_1);
  sc_2 = __ldlu(sc_1);
  sc_2 = __ldcv(sc_1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwb call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *sc_1 = sc_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *sc_1 = sc_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *sc_1 = sc_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwt call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *sc_1 = sc_2;
  __stwb(sc_1, sc_2);
  __stcg(sc_1, sc_2);
  __stcs(sc_1, sc_2);
  __stwt(sc_1, sc_2);

  uchar2 *uc2_1;
  uchar2 uc2_2;
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: uc2_2 = *uc2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldca call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: uc2_2 = *uc2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: uc2_2 = *uc2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldlu call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: uc2_2 = *uc2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcv call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: uc2_2 = *uc2_1;
  uc2_2 = __ldcg(uc2_1);
  uc2_2 = __ldca(uc2_1);
  uc2_2 = __ldcs(uc2_1);
  uc2_2 = __ldlu(uc2_1);
  uc2_2 = __ldcv(uc2_1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwb call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *uc2_1 = uc2_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *uc2_1 = uc2_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *uc2_1 = uc2_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwt call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *uc2_1 = uc2_2;
  __stwb(uc2_1, uc2_2);
  __stcg(uc2_1, uc2_2);
  __stcs(uc2_1, uc2_2);
  __stwt(uc2_1, uc2_2);

  uchar4 *uc4_1;
  uchar4 uc4_2;
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: uc4_2 = *uc4_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldca call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: uc4_2 = *uc4_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: uc4_2 = *uc4_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldlu call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: uc4_2 = *uc4_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcv call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: uc4_2 = *uc4_1;
  uc4_2 = __ldcg(uc4_1);
  uc4_2 = __ldca(uc4_1);
  uc4_2 = __ldcs(uc4_1);
  uc4_2 = __ldlu(uc4_1);
  uc4_2 = __ldcv(uc4_1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwb call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *uc4_1 = uc4_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *uc4_1 = uc4_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *uc4_1 = uc4_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwt call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *uc4_1 = uc4_2;
  __stwb(uc4_1, uc4_2);
  __stcg(uc4_1, uc4_2);
  __stcs(uc4_1, uc4_2);
  __stwt(uc4_1, uc4_2);

  uint2 *ui2_1;
  uint2 ui2_2;
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ui2_2 = *ui2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldca call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ui2_2 = *ui2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ui2_2 = *ui2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldlu call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ui2_2 = *ui2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcv call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ui2_2 = *ui2_1;
  ui2_2 = __ldcg(ui2_1);
  ui2_2 = __ldca(ui2_1);
  ui2_2 = __ldcs(ui2_1);
  ui2_2 = __ldlu(ui2_1);
  ui2_2 = __ldcv(ui2_1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwb call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *ui2_1 = ui2_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *ui2_1 = ui2_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *ui2_1 = ui2_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwt call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *ui2_1 = ui2_2;
  __stwb(ui2_1, ui2_2);
  __stcg(ui2_1, ui2_2);
  __stcs(ui2_1, ui2_2);
  __stwt(ui2_1, ui2_2);

  uint4 *ui4_1;
  uint4 ui4_2;
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ui4_2 = *ui4_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldca call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ui4_2 = *ui4_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ui4_2 = *ui4_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldlu call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ui4_2 = *ui4_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcv call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ui4_2 = *ui4_1;
  ui4_2 = __ldcg(ui4_1);
  ui4_2 = __ldca(ui4_1);
  ui4_2 = __ldcs(ui4_1);
  ui4_2 = __ldlu(ui4_1);
  ui4_2 = __ldcv(ui4_1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwb call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *ui4_1 = ui4_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *ui4_1 = ui4_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *ui4_1 = ui4_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwt call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *ui4_1 = ui4_2;
  __stwb(ui4_1, ui4_2);
  __stcg(ui4_1, ui4_2);
  __stcs(ui4_1, ui4_2);
  __stwt(ui4_1, ui4_2);

  ulonglong2 *ull2_1;
  ulonglong2 ull2_2;
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ull2_2 = *ull2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldca call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ull2_2 = *ull2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ull2_2 = *ull2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldlu call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ull2_2 = *ull2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcv call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ull2_2 = *ull2_1;
  ull2_2 = __ldcg(ull2_1);
  ull2_2 = __ldca(ull2_1);
  ull2_2 = __ldcs(ull2_1);
  ull2_2 = __ldlu(ull2_1);
  ull2_2 = __ldcv(ull2_1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwb call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *ull2_1 = ull2_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *ull2_1 = ull2_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *ull2_1 = ull2_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwt call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *ull2_1 = ull2_2;
  __stwb(ull2_1, ull2_2);
  __stcg(ull2_1, ull2_2);
  __stcs(ull2_1, ull2_2);
  __stwt(ull2_1, ull2_2);

  unsigned char *uc_1;
  unsigned char uc_2;
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: uc_2 = *uc_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldca call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: uc_2 = *uc_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: uc_2 = *uc_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldlu call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: uc_2 = *uc_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcv call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: uc_2 = *uc_1;
  uc_2 = __ldcg(uc_1);
  uc_2 = __ldca(uc_1);
  uc_2 = __ldcs(uc_1);
  uc_2 = __ldlu(uc_1);
  uc_2 = __ldcv(uc_1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwb call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *uc_1 = uc_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *uc_1 = uc_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *uc_1 = uc_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwt call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *uc_1 = uc_2;
  __stwb(uc_1, uc_2);
  __stcg(uc_1, uc_2);
  __stcs(uc_1, uc_2);
  __stwt(uc_1, uc_2);

  unsigned int *ui_1;
  unsigned int ui_2;
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ui_2 = *ui_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldca call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ui_2 = *ui_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ui_2 = *ui_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldlu call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ui_2 = *ui_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcv call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ui_2 = *ui_1;
  ui_2 = __ldcg(ui_1);
  ui_2 = __ldca(ui_1);
  ui_2 = __ldcs(ui_1);
  ui_2 = __ldlu(ui_1);
  ui_2 = __ldcv(ui_1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwb call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *ui_1 = ui_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *ui_1 = ui_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *ui_1 = ui_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwt call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *ui_1 = ui_2;
  __stwb(ui_1, ui_2);
  __stcg(ui_1, ui_2);
  __stcs(ui_1, ui_2);
  __stwt(ui_1, ui_2);

  unsigned long *ul_1;
  unsigned long ul_2;
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ul_2 = *ul_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldca call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ul_2 = *ul_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ul_2 = *ul_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldlu call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ul_2 = *ul_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcv call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ul_2 = *ul_1;
  ul_2 = __ldcg(ul_1);
  ul_2 = __ldca(ul_1);
  ul_2 = __ldcs(ul_1);
  ul_2 = __ldlu(ul_1);
  ul_2 = __ldcv(ul_1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwb call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *ul_1 = ul_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *ul_1 = ul_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *ul_1 = ul_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwt call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *ul_1 = ul_2;
  __stwb(ul_1, ul_2);
  __stcg(ul_1, ul_2);
  __stcs(ul_1, ul_2);
  __stwt(ul_1, ul_2);

  unsigned long long *ull_1;
  unsigned long long ull_2;
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ull_2 = *ull_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldca call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ull_2 = *ull_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ull_2 = *ull_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldlu call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ull_2 = *ull_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcv call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: ull_2 = *ull_1;
  ull_2 = __ldcg(ull_1);
  ull_2 = __ldca(ull_1);
  ull_2 = __ldcs(ull_1);
  ull_2 = __ldlu(ull_1);
  ull_2 = __ldcv(ull_1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwb call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *ull_1 = ull_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *ull_1 = ull_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *ull_1 = ull_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwt call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *ull_1 = ull_2;
  __stwb(ull_1, ull_2);
  __stcg(ull_1, ull_2);
  __stcs(ull_1, ull_2);
  __stwt(ull_1, ull_2);

  unsigned short *us_1;
  unsigned short us_2;
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: us_2 = *us_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldca call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: us_2 = *us_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: us_2 = *us_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldlu call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: us_2 = *us_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcv call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: us_2 = *us_1;
  us_2 = __ldcg(us_1);
  us_2 = __ldca(us_1);
  us_2 = __ldcs(us_1);
  us_2 = __ldlu(us_1);
  us_2 = __ldcv(us_1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwb call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *us_1 = us_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *us_1 = us_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *us_1 = us_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwt call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *us_1 = us_2;
  __stwb(us_1, us_2);
  __stcg(us_1, us_2);
  __stcs(us_1, us_2);
  __stwt(us_1, us_2);

  ushort2 *us2_1;
  ushort2 us2_2;
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: us2_2 = *us2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldca call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: us2_2 = *us2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: us2_2 = *us2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldlu call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: us2_2 = *us2_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcv call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: us2_2 = *us2_1;
  us2_2 = __ldcg(us2_1);
  us2_2 = __ldca(us2_1);
  us2_2 = __ldcs(us2_1);
  us2_2 = __ldlu(us2_1);
  us2_2 = __ldcv(us2_1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwb call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *us2_1 = us2_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *us2_1 = us2_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *us2_1 = us2_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwt call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *us2_1 = us2_2;
  __stwb(us2_1, us2_2);
  __stcg(us2_1, us2_2);
  __stcs(us2_1, us2_2);
  __stwt(us2_1, us2_2);

  ushort4 *us4_1;
  ushort4 us4_2;
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: us4_2 = *us4_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldca call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: us4_2 = *us4_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: us4_2 = *us4_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldlu call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: us4_2 = *us4_1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcv call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: us4_2 = *us4_1;
  us4_2 = __ldcg(us4_1);
  us4_2 = __ldca(us4_1);
  us4_2 = __ldcs(us4_1);
  us4_2 = __ldlu(us4_1);
  us4_2 = __ldcv(us4_1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwb call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *us4_1 = us4_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *us4_1 = us4_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *us4_1 = us4_2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwt call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *us4_1 = us4_2;
  __stwb(us4_1, us4_2);
  __stcg(us4_1, us4_2);
  __stcs(us4_1, us4_2);
  __stwt(us4_1, us4_2);
}

#define DEV_INLINE __device__ __forceinline__
__device__ __constant__ uint2 const keccak_round_constants[4] = {
    { 0x80008081, 0x80000000 }, { 0x00008080, 0x80000000 }, { 0x80000001, 0x00000000 }, { 0x80008008, 0x80000000 }
};


// CHECK:namespace dpct_operator_overloading {
// CHECK:inline sycl::uint2 &operator^=(sycl::uint2 &v, const sycl::uint2 &v2) {
// CHECK:  return v;
// CHECK:}
// CHECK:}  // namespace dpct_operator_overloading
__host__ __device__ inline uint2 &operator^=(uint2 &v, const uint2 &v2) {
  return v;
}


DEV_INLINE void SHA3_512(uint2* s) {
    int i;
  // CHECK:    dpct_operator_overloading::operator^=(s[0] , LDG(keccak_round_constants[i]));
  // CHECK-NEXT:    LDG(keccak_round_constants[23]);
    s[0] ^= LDG(keccak_round_constants[i]);
    LDG(keccak_round_constants[23]);
}
