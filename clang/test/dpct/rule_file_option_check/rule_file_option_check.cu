// RUN: cd %T
// RUN: cp %S/rule1.yaml %S/rule2.yaml .
// RUN: c2s --out-root out %s --cuda-include-path="%cuda-path/include" --rule-file=rule1.yaml  --rule-file=rule2.yaml
// RUN: c2s --out-root out %s --cuda-include-path="%cuda-path/include"  2>output.txt || true
// RUN: grep "\-\-rule-file=\".*rule1.yaml\" \-\-rule-file=\".*rule2.yaml\"" output.txt | wc -l > wc_output.txt
// RUN: FileCheck %s --match-full-lines --input-file wc_output.txt
// RUN: rm -rf out

// CHECK: 1

int main() {
  float2 f2;
  return 0;
}