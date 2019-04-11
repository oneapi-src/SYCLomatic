// RUN: rm  %T/header_migration_merge.h.yaml -f
// RUN: syclct -out-root %T %s  -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: syclct -out-root %T %s  -- -x cuda --cuda-host-only --cuda-path=%cuda-path -DMACROA
// RUN: FileCheck --input-file %T/header_migration_merge.sycl.cpp --match-full-lines %s
// RUN: FileCheck --input-file %T/header_migration_merge.h --match-full-lines %S/header_migration_merge.h

#include "header_migration_merge.h"
#include <stdio.h>

//CHECK: void helloGPU() {
__global__ void helloGPU() {
  printf("hello");
}
