// UNSUPPORTED: system-windows
// RUN: auto-compare --cuda-log %S/cuda.json --sycl %S/sycl.json &> %T/compare_log
// FileCheck --input-file %T/compare_log --match-full-lines %s
