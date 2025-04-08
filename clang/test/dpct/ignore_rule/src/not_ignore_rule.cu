// RUN: dpct --out-root %T %s --cuda-include-path="%cuda-path/include" --rule-file %S/rule2.yaml 2>&1 | tee %T/output2.txt
// RUN: FileCheck --input-file %T/not_ignore_rule.dp.cpp --match-full-lines %s
// RUN: FileCheck --input-file %T/output2.txt --match-full-lines %S/output2_ref.txt

#include "../not_ignore_rule.h"

// CHECK: void foo(Tensor a) {
// CHECK-NEXT:   a.is_xpu();
// CHECK-NEXT: }
void foo(Tensor a) {
  a.is_cuda();
}
