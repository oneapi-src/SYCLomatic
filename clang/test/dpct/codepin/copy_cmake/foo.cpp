// UNSUPPORTED: system-windows
// RUN: dpct --in-root %S -out-root %T/foo  --cuda-include-path="%cuda-path/include" %s --enable-codepin  -- -std=c++14  -x cuda --cuda-host-only || true

// RUN: cd %T/foo_codepin_cuda
// RUN: ls > default2.log
// RUN: ls test >> default2.log
// RUN: FileCheck --input-file default2.log --match-full-lines %T/foo_codepin_cuda/foo.cpp -check-prefix=DEFAULT
// DEFAULT: CMakeLists.txt
// DEFAULT: TestCMAKE.cmake
// DEFAULT: test.cmake


// RUN: echo "begin" > %T/diff.txt
// RUN: diff --strip-trailing-cr %S/cmake_expected %T/foo_codepin_cuda/CMakeLists.txt >> %T/diff.txt
// RUN: diff --strip-trailing-cr %S/expected.cmake %T/foo_codepin_cuda/TestCMAKE.cmake >> %T/diff.txt
// RUN: echo "end" >> %T/diff.txt
// RUN: FileCheck --input-file %T/diff.txt --check-prefix=CHECK %s
// CHECK: begin
// CHECK-NEXT:end

#include <cuda.h>

int main() {
  return 0;
}
