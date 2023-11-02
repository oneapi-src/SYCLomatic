#define ATOMIC_ADD(x, v)  atomicAdd(&x, v);

// RUN: dpct --format-range=none --out-root %T/unused_atomic_macro %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file %T/unused_atomic_macro/unused_atomic_macro.dp.cpp --match-full-lines %s

// CHECK: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT:  /*
// CHECK-NEXT: DPCT1058:{{[0-9]+}}: "atomicAdd" is not migrated because it is not called in the code.
// CHECK-NEXT: */
// CHECK-NEXT: #define ATOMIC_ADD(x, v) atomicAdd(&x, v);

void foo() {}
