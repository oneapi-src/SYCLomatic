// RUN: c2s --format-range=none -out-root %T/duplicate2 %S/duplicate2.cu %s --cuda-include-path="%cuda-path/include" -- -std=c++14  -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/duplicate2/duplicate1.dp.cpp
// RUN: FileCheck %S/duplicate2.cu --match-full-lines --input-file %T/duplicate2/duplicate2.dp.cpp

__constant__ int ca[32];

// CHECK: void kernel(int *i, int *ca) {
__global__ void kernel(int *i) {
  ca[0] = *i;
}