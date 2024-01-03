// ------ prepare test directory
// RUN: rm -rf %T/dpct_output
// RUN: cd %T
// RUN: cp %S/one.cu one.cu
// RUN: cp %S/compile_commands.json compile_commands.json
// RUN: dpct -p=%T/compile_commands.json -out-root=%T/dpct_output
// RUN: FileCheck --match-full-lines --input-file %T/dpct_output/one.dp.cpp %T/one.cu