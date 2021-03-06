// RUN: dpct --check-unicode-security --format-range=none -out-root %T/confusable_identifier %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/confusable_identifier/confusable_identifier.dp.cpp --match-full-lines %s

// CHECK: /*
// CHECK: DPCT1095:{{[0-9]+}}: The identifier "fo" is confusable with another identifier "ðo".
// CHECK: */
int fo;
// CHECK: /*
// CHECK: DPCT1095:{{[0-9]+}}: The identifier "ðo"  is confusable with another identifier "fo".
// CHECK: */
int ðo;

void no() {
  int ðoo;
}

void worry() {
  int foo;
}
// CHECK: /*
// CHECK: DPCT1095:{{[0-9]+}}: The identifier "ði"  is confusable with another identifier "fi".
// CHECK: */
int ði;
// CHECK: /*
// CHECK: DPCT1095:{{[0-9]+}}: The identifier "fi" is confusable with another identifier "ði".
// CHECK: */
int fi;
