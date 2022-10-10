// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.2, cuda-11.7
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.2, v11.7
// ------ make test directory
// RUN: cd %T
// RUN: rm -rf header_guard_test
// RUN: mkdir  header_guard_test
// RUN: cd     header_guard_test
//
// ------ use pre-existing tests to trigger creation of custom header files
// RUN: cp %S/thrust-iterators.cu        .
// RUN: cp %S/cublas_curandInMacro.cu    .
// RUN: cp %S/nccl.cu                    .
//
// ------ create custom header files
// RUN:  dpct thrust-iterators.cu cublas_curandInMacro.cu nccl.cu --use-custom-helper=api --cuda-include-path="%cuda-path/include"
// RUN:  grep ifndef dpct_output/include/dpct/*.hpp > header_guards.txt
//
// ------ ensure header guard names in custom header files match the names in the standard header files
// RUN: FileCheck --input-file header_guards.txt %s
//
// ------ cleanup test directory
// RUN: cd ..
// RUN: rm -rf ./gen-build-script

// CHECK: dpct_output/include/dpct/blas_utils.hpp:#ifndef __DPCT_BLAS_UTILS_HPP__
// CHECK: dpct_output/include/dpct/ccl_utils.hpp:#ifndef __DPCT_CCL_UTILS_HPP__
// CHECK: dpct_output/include/dpct/device.hpp:#ifndef __DPCT_DEVICE_HPP__
// CHECK: dpct_output/include/dpct/dpct.hpp:#ifndef __DPCT_HPP__
// CHECK: dpct_output/include/dpct/dpl_utils.hpp:#ifndef __DPCT_DPL_UTILS_HPP__
// CHECK: dpct_output/include/dpct/memory.hpp:#ifndef __DPCT_MEMORY_HPP__
// CHECK: dpct_output/include/dpct/util.hpp:#ifndef __DPCT_UTIL_HPP__

void foo() {
}
