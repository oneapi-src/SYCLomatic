// RUN: cd %T
// RUN: cat %S/compile_commands.json > %T/compile_commands.json
// RUN: cat %S/test.cu > %T/test.cu

// RUN: dpct  -p=%T  -in-root=%T -out-root=%T/out -format-range=none --cuda-include-path="%cuda-path/include"


// RUN: grep "<dpct/dpct.hpp>" %T/out/MainSourceFiles.yaml | wc -l > %T/wc_output.txt || true
// RUN: FileCheck --input-file %T/wc_output.txt --match-full-lines %s


// CHECK: 0

#include<iostream>

int main() {
      unsigned int test = (unsigned int) log2(2.0);
}
