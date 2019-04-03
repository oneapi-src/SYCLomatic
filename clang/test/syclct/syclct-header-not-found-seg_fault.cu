// RUN: syclct -no-stop-on-err -out-root %T %s -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck %s --match-full-lines --input-file %T/syclct-header-not-found-seg_fault.sycl.cpp

#include "NonExistantHeaderFile.h"

// CHECK: void hello();
__global__ void hello();

