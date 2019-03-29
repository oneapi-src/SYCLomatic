// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck %s --match-full-lines --input-file %T/syclct-header-not-found-seg_fault.sycl.cpp

//Will enable this case after https://jira.devtools.intel.com/browse/CTST-545 has been fixed.
//#include "NonExistantHeaderFile.h"

// CHECK: void hello();
__global__ void hello();

