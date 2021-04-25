// RUN: cp %S/MainSourceFiles.yaml %T
// RUN: dpct --format-range=none --out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only > %T/output.txt
// RUN: FileCheck --input-file %T/output.txt --match-full-lines %S/output_ref.txt
// RUN: rm -rf %T/*

int main() {
  float *d_A;
  cudaMalloc(&d_A, 10 * sizeof(float));
  return 0;
}
