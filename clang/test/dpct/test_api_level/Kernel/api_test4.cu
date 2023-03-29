// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Kernel/api_test4_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Kernel/api_test4_out/MainSourceFiles.yaml | wc -l > %T/Kernel/api_test4_out/count.txt
// RUN: FileCheck --input-file %T/Kernel/api_test4_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Kernel/api_test4_out

// CHECK: 29
// TEST_FEATURE: Kernel_kernel_library
// TEST_FEATURE: Kernel_get_image_wrapper

#include "cuda.h"

int main() {
  CUmodule module;
  cuModuleLoad(&module,"kernel_library.ptx");
  CUmodule M;
  CUtexref tex;
  cuModuleGetTexRef(&tex, M, "tex");
}
