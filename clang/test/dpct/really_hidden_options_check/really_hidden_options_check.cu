// RUN: cd %T
// RUN: dpct --out-root out %s --cuda-include-path="%cuda-path/include"
// RUN: grep "NoUseGenericSpace" out/MainSourceFiles.yaml | wc -l > wc_output.txt || true
// RUN: FileCheck %s --match-full-lines --input-file wc_output.txt

// CHECK: 0
int main() {
  float2 f2;
  return 0;
}