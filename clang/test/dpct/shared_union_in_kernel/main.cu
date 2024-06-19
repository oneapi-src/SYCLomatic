// RUN: cp %S/* .
// RUN: dpct --format-range=none -p=. -out-root=%T --cuda-include-path="%cuda-path/include" -- -x -std=c++14
// RUN: FileCheck --input-file %T/common.dp.hpp --match-full-lines %S/common.cuh
// RUN: %if build_lit %{icpx -c -fsycl %T/main.dp.cpp %T/test.dp.cpp %}

#include "common.cuh"

int main() { return 0; }
