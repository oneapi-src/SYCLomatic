// RUN: cat %s > %T/user_define_rule_header_order1.cu
// RUN: cat %S/user_define_rule_header_order1.yaml > %T/user_define_rule_header_order1.yaml
// RUN: cat %S/user_define_rule_header_order1.h > %T/user_define_rule_header_order1.h
// RUN: cd %T
// RUN: rm -rf %T/user_define_rule_header_order1_output
// RUN: mkdir %T/user_define_rule_header_order1_output
// RUN: dpct -format-range=none -out-root %T/user_define_rule_header_order1_output user_define_rule_header_order1.cu --cuda-include-path="%cuda-path/include" --usm-level=none --rule-file=user_define_rule_header_order1.yaml -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/user_define_rule_header_order1_output/user_define_rule_header_order1.dp.cpp --match-full-lines user_define_rule_header_order1.cu

// CHECK: #include <oneapi/dpl/execution>
// CHECK: #include <oneapi/dpl/algorithm>
// CHECK: #include <sycl/sycl.hpp>
#include <cub/cub.cuh>
#include <stddef.h>
#include "user_define_rule_header_order1.h"
