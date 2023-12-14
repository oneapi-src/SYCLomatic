// RUN: dpct --in-root %S --out-root %T %S/a.cu %S/a_kernel.cu %S/a.h  --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file %T/a_kernel.dp.cpp --match-full-lines %S/a_kernel.cu
// RUN: %if build_lit %{icpx -c -fsycl %T/a_kernel.dp.cpp -o %T/a_kernel.dp.o %}

#include "a.h"
#include "a_kernel.cu"
