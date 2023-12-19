// RUN: dpct --format-range=none -out-root %T/dpct-header-not-found-seg_fault %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/dpct-header-not-found-seg_fault/dpct-header-not-found-seg_fault.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/dpct-header-not-found-seg_fault/dpct-header-not-found-seg_fault.dp.cpp -o %T/dpct-header-not-found-seg_fault/dpct-header-not-found-seg_fault.dp.o %}

#include "NonExistantHeaderFile.h"

// CHECK: void hello();
__global__ void hello();


