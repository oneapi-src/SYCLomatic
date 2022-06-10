// UNSUPPORTED: -windows-

// RUN: cat %S/compile_commands.json > %T/compile_commands.json
// RUN: cat %S/one.cu > %T/one.cu
// RUN: cat %S/three.cpp > %T/three.cpp
// RUN: cat %S/two.cpp > %T/two.cpp
// RUN: cd %T

// RUN: dpct  -p=%T  -in-root=%T -out-root=%T/out -gen-build-script --cuda-include-path="%cuda-path/include"
// RUN: cat %S/Makefile.dpct.ref  >%T/Makefile.dpct.check
// RUN: cat %T/out/Makefile.dpct >> %T/Makefile.dpct.check
// RUN: FileCheck --match-full-lines --input-file %T/Makefile.dpct.check %T/Makefile.dpct.check

#include <iostream>
#include <thrust/device_vector.h>

int main() {
  thrust::device_vector<int> a;
  return 0;
}
