// RUN: cd %T
// RUN: cat %s > test_yaml.cu
// RUN: cat %S/a_test_yaml.h > a_test_yaml.h
// RUN: dpct --out-root=test_yaml_out %s --cuda-include-path="%cuda-path/include"  -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/test_yaml_out/a_test_yaml.h.yaml --match-full-lines %s
// RUN: FileCheck --input-file %T/test_yaml_out/a_test_yaml.h --match-full-lines %S/a_test_yaml.h
// RUN: rm -rf ./test_yaml_out

// CHECK: ---
#include "a_test_yaml.h"

