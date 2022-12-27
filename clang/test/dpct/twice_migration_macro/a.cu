// RUN: dpct --in-root %S --out-root %T %S/a.cu %S/a_kernel.cu %S/a.h  --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file %T/a_kernel.dp.cpp --match-full-lines %S/a_kernel.cu

#include "a.h"
#include "a_kernel.cu"
