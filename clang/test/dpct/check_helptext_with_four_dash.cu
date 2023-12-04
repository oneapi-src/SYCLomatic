// RUN: cd %T
// RUN: dpct -- -help > helptext1.txt
// RUN: echo "This line is inserted for checking END." >> helptext1.txt
// RUN: cat %S/check_helptext_with_four_dash_ref.txt  >%T/check_helptext_with_four_dash_ref.txt
// RUN: FileCheck --match-full-lines --input-file %T/helptext1.txt %T/check_helptext_with_four_dash_ref.txt

// RUN: dpct -- --help > helptext2.txt
// RUN: echo "This line is inserted for checking END." >> helptext2.txt
// RUN: FileCheck --match-full-lines --input-file %T/helptext2.txt %T/check_helptext_with_four_dash_ref.txt

// RUN: cat %S/check_helptext_with_four_dash_ref1.txt  >%T/check_helptext_with_four_dash_ref1.txt
// RUN: dpct -- --help-hidden > helptext3.txt
// RUN: echo "This line is inserted for checking END." >> helptext3.txt
// RUN: FileCheck --match-full-lines --input-file %T/helptext3.txt %T/check_helptext_with_four_dash_ref1.txt