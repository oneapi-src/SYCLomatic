// RUN: rm  %T/header_migration_merge/header_migration_merge.h.yaml -f
// RUN: dpct --format-range=none -out-root %T/header_migration_merge %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: dpct --format-range=none -out-root %T/header_migration_merge %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -DMACROA
// RUN: FileCheck --input-file %T/header_migration_merge/header_migration_merge.dp.cpp --match-full-lines %s
// RUN: FileCheck --input-file %T/header_migration_merge/header_migration_merge.h --match-full-lines %S/header_migration_merge.h

#include "header_migration_merge.h"
#include <stdio.h>

//CHECK: void helloGPU() {
__global__ void helloGPU() {
}
