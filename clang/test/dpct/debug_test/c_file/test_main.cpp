// RUN: dpct --enable-codepin --out-root %T/debug_test/c_file %S/test_main.cpp %S/test_cuda.cu --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %S/test_main.cpp --match-full-lines --input-file %T/debug_test/c_file_codepin_sycl/test_main.cpp -check-prefix=SYCL
// RUN: FileCheck %S/test_cuda.cu --match-full-lines --input-file %T/debug_test/c_file_codepin_sycl/test_cuda.dp.cpp -check-prefix=SYCL
// RUN: FileCheck %S/test.h --match-full-lines --input-file %T/debug_test/c_file_codepin_sycl/test.h -check-prefix=SYCL
// RUN: FileCheck %S/test_main.cpp --match-full-lines --input-file %T/debug_test/c_file_codepin_cuda/test_main.cpp -check-prefix=CUDA
// RUN: FileCheck %S/test_cuda.cu --match-full-lines --input-file %T/debug_test/c_file_codepin_cuda/test_cuda.cu -check-prefix=CUDA
// RUN: FileCheck %S/test.h --match-full-lines --input-file %T/debug_test/c_file_codepin_cuda/test.h -check-prefix=CUDA
// RUN: %if build_lit %{icpx -fsycl %T/debug_test/c_file_codepin_sycl/test_main.cpp %T/debug_test/c_file_codepin_sycl/test_cuda.dp.cpp -o %T/debug_test/c_file_codepin_sycl/c_file.run %}
// RUN: rm -rf %T/debug_test/c_file_codepin_sycl
// RUN: rm -rf %T/debug_test/c_file_debug_cuda

// SYCL: #include "test.h"
// SYCL-NEXT: int main() {
// SYCL-NEXT:   foo();
// SYCL-NEXT:   return 0;
// SYCL-NEXT: }
// CUDA: #include "test.h"
// CUDA-NEXT: int main() {
// CUDA-NEXT:   foo();
// CUDA-NEXT:   return 0;
// CUDA-NEXT: }
#include "test.h"
int main() {
  foo();
  return 0;
}
