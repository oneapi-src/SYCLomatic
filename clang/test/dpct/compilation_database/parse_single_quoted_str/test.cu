// RUN: cd %T
// RUN: cat %S/compile_commands.json > %T/compile_commands.json
// RUN: cat %s > %T/test.cu
// RUN: cat %S/test_isolate.cu > %T/test_isolate.cu
// RUN: dpct --format-range=none -in-root=%T  -out-root=%T/out -p ./ --format-range=none --cuda-include-path="%cuda-path/include" --process-all
// RUN: FileCheck %s --match-full-lines --input-file %T/out/test.dp.cpp
// RUN: %if build_lit %{icpx -DNAMD="\"3.0b3\"" -c -fsycl %T/out/test.dp.cpp -o %T/out/test.dp.o %}
// RUN: FileCheck %S/test_isolate.cu --match-full-lines --input-file %T/out/test_isolate.dp.cpp

// CHECK:  #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <iostream>
#include <cuda_runtime.h>
#include <iostream>
int main() {
  std::cout << NAMD;
  return 0;
}
