// RUN: c2s --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/LibCommonUtils/api_test1_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/LibCommonUtils/api_test1_out/MainSourceFiles.yaml | wc -l > %T/LibCommonUtils/api_test1_out/count.txt
// RUN: FileCheck --input-file %T/LibCommonUtils/api_test1_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/LibCommonUtils/api_test1_out

// CHECK: 2
// TEST_FEATURE: LibCommonUtils_version_field

int main() {
  libraryPropertyType ver = MAJOR_VERSION;
  return 0;
}
