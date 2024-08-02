// UNSUPPORTED: system-windows
// RUN: dpct --in-root %S -out-root %T/foo  --cuda-include-path="%cuda-path/include" --enable-codepin --migrate-build-script=CMake -- -std=c++14  -x cuda --cuda-host-only || true
// RUN: cp %s %T/foo_codepin_sycl
// RUN: cd %T/foo_codepin_sycl
// RUN: ls  > default.log
// RUN: FileCheck --input-file default.log --match-full-lines %T/foo_codepin_sycl/foo.cpp -check-prefix=DEFAULT
// RUN: cd %T/foo_codepin_cuda
// RUN: ls > default2.log
// RUN: FileCheck --input-file default2.log --match-full-lines %T/foo_codepin_sycl/foo.cpp -check-prefix=DEFAULT
// DEFAULT: CMakeLists.txt

#include <iostream>

int main() {
  return 0;
}
