// RUN: rm -rf %T/included_with_error_output
// RUN: mkdir %T/included_with_error_output
// RUN: cat %s > %T/include_parse_error.cu
// RUN: cat %S/included_with_parse_error.cu > %T/included_with_parse_error.cu
// RUN: cd %T
// RUN: c2s --format-range=none -out-root %T/included_with_error_output include_parse_error.cu included_with_parse_error.cu --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/included_with_error_output/included_with_parse_error.dp.cpp --match-full-lines included_with_parse_error.cu

#define CALL(x) x
//CHECK: #include"included_with_parse_error.dp.cpp"
#include"included_with_parse_error.cu"
