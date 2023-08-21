
#define POW(B, E) pow(B, E)
__device__ void foo() {
  POW(2.5, 3.1);
  POW(2.5, 3);
}

// RUN: dpct --format-range=none -out-root %T/warning_insert_location %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/warning_insert_location/warning_insert_location.dp.cpp

// CHECK: /*
// CHECK-NEXT: DPCT1064:{{[0-9]+}}: Migrated pow call is used in a macro/template definition and may not be valid for all macro/template uses. Adjust the code.
// CHECK-NEXT: */
// CHECK-NEXT: #define POW(B, E) sycl::pow<double>(B, E)
// CHECK-NEXT: void foo() {
// CHECK-NEXT:   POW(2.5, 3.1);
// CHECK-NEXT:   POW(2.5, 3);
// CHECK-NEXT: }
