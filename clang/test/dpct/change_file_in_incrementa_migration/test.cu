// RUN: cp %S/* .
// RUN: dpct --process-all --in-root=. --out-root=%T --cuda-include-path="%cuda-path/include" -- -DAAA
// RUN: FileCheck --input-file %T/test.dp.cpp --match-full-lines %S/migrate_ref1.txt
// RUN: echo "//" >> test.cu
// RUN: dpct --process-all --in-root=. --out-root=%T 2> %T/output1.txt --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file %T/test.dp.cpp --match-full-lines %S/migrate_ref2.txt
// RUN: FileCheck --input-file %T/output1.txt --match-full-lines %S/output_ref1.txt
// RUN: echo "//" >> test.cuh
// RUN: dpct --process-all --in-root=. --out-root=%T 2> %T/output2.txt --cuda-include-path="%cuda-path/include" -- -DAAA
// RUN: FileCheck --input-file %T/test.dp.cpp --match-full-lines %S/migrate_ref3.txt
// RUN: FileCheck --input-file %T/output2.txt --match-full-lines %S/output_ref2.txt
// RUN: rm -rf %T/*

#include "test.cuh"
#ifdef AAA
__constant__ const float bbb = 1.0;
#else
__constant__ const float ccc = 1.0;
#endif
int main() {
  return 0;
}
