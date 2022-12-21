// UNSUPPORTED: -windows-

// RUN: cp %S/* .
// RUN: dpct -p=%S --out-root=%T --cuda-include-path="%cuda-path/include"
// RUN: ls Output > default.log
// RUN: FileCheck --input-file default.log --match-full-lines %T/main.dp.cpp -check-prefix=DEFAULT
// DEFAULT: main.dp.cpp
// DEFAULT: test.cpp.dp.cpp
// DEFAULT: test.dp.hpp
// DEFAULT: test.h

// RUN: rm %T/*
// RUN: dpct -p=%S --out-root=%T --cuda-include-path="%cuda-path/include" --change-filename-extension=".cu"
// RUN: ls Output > cu.log
// RUN: FileCheck --input-file cu.log --match-full-lines %T/main.dp.cpp -check-prefix=CU
// CU: main.dp.cpp
// CU: test.cpp
// CU: test.cuh
// CU: test.h

// RUN: rm %T/*
// RUN: dpct -p=%S --out-root=%T --cuda-include-path="%cuda-path/include" --change-filename-extension=".abc"
// RUN: ls Output > none.log
// RUN: FileCheck --input-file none.log --match-full-lines %T/main.cu -check-prefix=NONE
// NONE: main.cu
// NONE: test.cpp
// NONE: test.cuh
// NONE: test.h

#include "test.cuh"
#include "test.h"

int main() {
  f<<<1, 1>>>();
  return 0;
}
