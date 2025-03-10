// REQUIRES: system-windows

// RUN: rm -rf %T/help_option_check && mkdir -p %T/help_option_check
// RUN: cd %T/help_option_check

// RUN: echo "begin" > %T/diff.txt
// RUN: dpct --help > output.txt
// RUN: diff --strip-trailing-cr %S/help_all.txt %T/help_option_check/output.txt >> %T/diff.txt
// RUN: dpct --help=basic > output.txt
// RUN: diff --strip-trailing-cr %S/help_basic.txt %T/help_option_check/output.txt >> %T/diff.txt
// RUN: dpct --help=advanced > output.txt
// RUN: diff --strip-trailing-cr %S/help_advanced.txt %T/help_option_check/output.txt >> %T/diff.txt
// RUN: echo "end" >> %T/diff.txt

// RUN: cat %T/diff.txt | FileCheck %s
// CHECK: begin
// CHECK-NEXT: end
