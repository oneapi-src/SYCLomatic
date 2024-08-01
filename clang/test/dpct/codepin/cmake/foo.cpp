// UNSUPPORTED: system-windows
// RUN: dpct --in-root %S -out-root %T/foo  --cuda-include-path="%cuda-path/include" --enable-codepin --migrate-build-script=CMake -- -std=c++14  -x cuda --cuda-host-only || true
// RUN: cp %s %T/foo_codepin_sycl
// RUN: cd %T/foo_codepin_sycl && mkdir build && cd build && cmake ..

#include <iostream>

int main() {
  return 0;
}
