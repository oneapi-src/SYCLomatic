// RUN: auto-compare --cuda-log %S/cuda.json --sycl %S/sycl.json &> %T/compare_log
// FileCheck --input-file %T/compare_log --match-full-lines %s

// CHECK: Comparison succeeded, no differences found.
