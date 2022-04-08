// RUN: c2s --check-unicode-security --format-range=none -out-root %T/confusable_identifier %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/confusable_identifier/confusable_identifier.dp.cpp --match-full-lines %s

// CHECK: /*
// CHECK: DPCT1095:{{[0-9]+}}: The identifier "fo" is confusable with another identifier "𝐟o".
// CHECK: */
int fo;
// CHECK: /*
// CHECK: DPCT1095:{{[0-9]+}}: The identifier "𝐟o"  is confusable with another identifier "fo".
// CHECK: */
int 𝐟o;

void no() {
  int 𝐟oo;
}

void worry() {
  int foo;
}
// CHECK: /*
// CHECK: DPCT1095:{{[0-9]+}}: The identifier "𝐟i"  is confusable with another identifier "fi".
// CHECK: */
int 𝐟i;
// CHECK: /*
// CHECK: DPCT1095:{{[0-9]+}}: The identifier "fi" is confusable with another identifier "𝐟i".
// CHECK: */
int fi;
