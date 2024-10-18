// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T/sleep %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/sleep/sleep.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST  %T/sleep/sleep.dp.cpp -o %T/sleep/sleep.dp.o %}

#ifndef NO_BUILD_TEST

__global__ static void sleep_kernel() {
  // CHECK: /*
  // CHECK-NEXT: DPCT1008:{{[0-9]+}}: __nanosleep function is not defined in SYCL. This is a hardware-specific feature. Consult with your hardware vendor to find a replacement.
  // CHECK-NEXT: */
  __nanosleep(1);
}

#endif