// RUN: dpct --out-root %T %s --cuda-include-path="%cuda-path/include" > %T/auto_deduce_output.txt 2>&1 || true
// RUN: grep "dpct internal error" %T/auto_deduce_output.txt | wc -l > %T/wc_auto_deduce_output.txt || true
// RUN: FileCheck %s --match-full-lines --input-file %T/wc_auto_deduce_output.txt
// RUN: rm -rf %T

// CHECK: 0

__device__ void test_auto() {

  auto tid = get_tid();
}
