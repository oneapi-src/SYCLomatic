// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Util/api_test9_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Util/api_test9_out/MainSourceFiles.yaml | wc -l > %T/Util/api_test9_out/count.txt
// RUN: FileCheck --input-file %T/Util/api_test9_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Util/api_test9_out

// CHECK: 15

// TEST_FEATURE: Util_make_index_sequence

int main() {
  cudaArray_t a42;
  return 0;
}
