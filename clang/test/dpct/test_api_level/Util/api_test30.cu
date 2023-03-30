// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Util/api_test30_out %s --cuda-include-path="%cuda-path/include"
// RUN: grep "IsCalled" %T/Util/api_test30_out/MainSourceFiles.yaml | wc -l > %T/Util/api_test30_out/count.txt
// RUN: FileCheck --input-file %T/Util/api_test30_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Util/api_test30_out

// CHECK: 2
// TEST_FEATURE: Math_min_max

__device__ void foo() {
  int i;
  max(i, i);
}
