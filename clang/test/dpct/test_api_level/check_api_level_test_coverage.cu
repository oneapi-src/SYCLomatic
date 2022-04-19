// UNSUPPORTED: -windows-
// RUN: c2s -out-root %T/check_api_level_test_coverage_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: bash %S/script.sh %S %T
// RUN: FileCheck --input-file %T/result.txt --match-full-lines %s


// CHECK: begin
// CHECK-NEXT: end


// This test is used to check if there is any new feaure requested in source code without related test added.
// In script.sh, it will grep all features used in source code and grep all features have been tested in lit.
// If the two results are same, then this case will pass. Otherwise, you need add new test in test_api_level folder.

float2 f2;