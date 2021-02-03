// RUN: dpct --format-range=none --usm-level=none -in-root %S -out-root %T/redefine_sycl_type %s %S/redefine_sycl_type_b.cu -extra-arg="-I %S" --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/redefine_sycl_type/redefine_sycl_type_a.c.dp.cpp
#include <stdio.h>
#include <unistd.h>
#include <fstream>
#include <stdint.h>

// CHECK: /*
// CHECK-NEXT: DPCT1077:{{[0-9]+}}: 'int2' redefines a standard SYCL type, which may cause conflicts.
// CHECK-NEXT: */
// CHECK-NEXT: #define int2 int32_t
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1077:{{[0-9]+}}: 'ulong4' redefines a standard SYCL type, which may cause conflicts.
// CHECK-NEXT: */
// CHECK-NEXT: #define ulong4 uint32_t
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1077:{{[0-9]+}}: 'uint4' redefines a standard SYCL type, which may cause conflicts.
// CHECK-NEXT: */
// CHECK-NEXT: #define uint4 uint32_t
#define int2 int32_t
#define ulong4 uint32_t
#define uint4 uint32_t
#include "redefine_sycl_type.h"

int main() {
  return 0;
}
