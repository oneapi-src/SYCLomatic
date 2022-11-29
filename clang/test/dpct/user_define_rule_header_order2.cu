// RUN: cat %s > %T/user_define_rule_header_order2.cu
// RUN: cat %S/user_define_rule_header_order2.yaml > %T/user_define_rule_header_order2.yaml
// RUN: cd %T
// RUN: rm -rf %T/user_define_rule_header_order2_output
// RUN: mkdir %T/user_define_rule_header_order2_output
// RUN: dpct -format-range=none -out-root %T/user_define_rule_header_order2_output user_define_rule_header_order2.cu --cuda-include-path="%cuda-path/include" --rule-file=user_define_rule_header_order2.yaml -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/user_define_rule_header_order2_output/user_define_rule_header_order2.dp.cpp --match-full-lines user_define_rule_header_order2.cu

// CHECK: #include <dpct/dnnl_utils.hpp>
// CHECK: #include <sycl/sycl.hpp>
// CHECK: #include <dpct/dpct.hpp>
// CHECK: #include <iostream>
// CHECK: #include <vector>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <vector>

template<cudnnDataType_t T>
struct dt_trait{
    typedef void type;
};
