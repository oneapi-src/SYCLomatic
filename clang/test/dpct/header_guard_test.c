// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.2
// ------ create custom header files
// RUN:  dpct --out-root %T/header_guard_test %S/thrust-iterators.cu %S/cublas_curandInMacro.cu %S/nccl.cu --use-custom-helper=api --cuda-include-path="%cuda-path/include"
// RUN:  grep --no-filename ifndef %T/header_guard_test/include/dpct/*.hpp > %T/header_guard_test/header_guards.txt
//
// ------ ensure header guard names in custom header files match the names in the standard header files
// RUN: FileCheck --input-file %T/header_guard_test/header_guards.txt %s

// CHECK: #ifndef __DPCT_BLAS_UTILS_HPP__
// CHECK: #ifndef __DPCT_CCL_UTILS_HPP__
// CHECK: #ifndef __DPCT_DEVICE_HPP__
// CHECK: #ifndef __DPCT_HPP__
// CHECK: #ifndef __DPCT_DPL_UTILS_HPP__
// CHECK: #ifndef __DPCT_LIB_COMMON_UTILS_HPP__
// CHECK: #ifndef __DPCT_MEMORY_HPP__
// CHECK: #ifndef __DPCT_UTIL_HPP__

void foo() {
}
