// RUN: c2s --format-range=none -out-root %T/c2s-header-not-found-seg_fault %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/c2s-header-not-found-seg_fault/c2s-header-not-found-seg_fault.dp.cpp

#include "NonExistantHeaderFile.h"

// CHECK: void hello();
__global__ void hello();


