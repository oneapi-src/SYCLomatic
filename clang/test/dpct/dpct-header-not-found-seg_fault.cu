// RUN: dpct -no-stop-on-err -out-root %T %s -- -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck %s --match-full-lines --input-file %T/dpct-header-not-found-seg_fault.dp.cpp

#include "NonExistantHeaderFile.h"

// CHECK: void hello();
__global__ void hello();

