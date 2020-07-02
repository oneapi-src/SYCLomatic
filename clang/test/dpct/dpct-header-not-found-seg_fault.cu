// RUN: dpct --format-range=none  -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/dpct-header-not-found-seg_fault.dp.cpp

#include "NonExistantHeaderFile.h"

// CHECK: void hello();
__global__ void hello();

