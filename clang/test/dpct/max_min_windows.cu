// RUN: dpct --format-range=none -out-root %T/max_min_windows %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/max_min_windows/max_min_windows.dp.cpp

#if defined(_WIN32) || defined(WIN32)
#include <Windows.h>
#endif

__global__ void test_max_min(void) {
  float a = 2.0, b = 3.0;

  // CHECK: float c = sycl::max(a, b);
  float c = max(a, b);

  // CHECK: float d = sycl::min(a, b);
  float d = min(a, b);
}

