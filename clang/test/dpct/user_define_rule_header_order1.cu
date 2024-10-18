// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.2, cuda-11.4
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.2, v11.4
// RUN: cat %s > %T/user_define_rule_header_order1.cu
// RUN: cat %S/user_define_rule_header_order1.yaml > %T/user_define_rule_header_order1.yaml
// RUN: cat %S/user_define_rule_header_order1.h > %T/user_define_rule_header_order1.h
// RUN: cd %T
// RUN: rm -rf %T/user_define_rule_header_order1_output
// RUN: mkdir %T/user_define_rule_header_order1_output
// RUN: dpct -format-range=none -out-root %T/user_define_rule_header_order1_output user_define_rule_header_order1.cu --cuda-include-path="%cuda-path/include" --usm-level=none --rule-file=user_define_rule_header_order1.yaml -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/user_define_rule_header_order1_output/user_define_rule_header_order1.dp.cpp --match-full-lines user_define_rule_header_order1.cu
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST  %T/user_define_rule_header_order1_output/user_define_rule_header_order1.dp.cpp -o %T/user_define_rule_header_order1_output/user_define_rule_header_orde1r.dp.o %}

// CHECK: #include <oneapi/dpl/execution>
// CHECK: #include <oneapi/dpl/algorithm>
#include <stddef.h>
#include "user_define_rule_header_order1.h"
