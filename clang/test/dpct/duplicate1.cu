// RUN: dpct --format-range=none -out-root %T/duplicate2 %S/duplicate2.cu %s --cuda-include-path="%cuda-path/include" -- -std=c++14  -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/duplicate2/duplicate1.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST  %T/duplicate2/duplicate1.dp.cpp -o %T/duplicate2/duplicate1.dp.o %}
// RUN: FileCheck %S/duplicate2.cu --match-full-lines --input-file %T/duplicate2/duplicate2.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST  %T/duplicate2/duplicate2.dp.cpp -o %T/duplicate2/duplicate2.dp.o %}
#ifndef  NO_BUILD_TEST
__constant__ int ca[32];

// CHECK: void kernel(int *i, int const *ca) {
__global__ void kernel(int *i) {
  ca[0] = *i;
}
#endif