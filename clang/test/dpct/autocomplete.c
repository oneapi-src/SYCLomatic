// UNSUPPORTED: -windows-

// RUN: dpct --autocomplete=
// RUN: dpct --autocomplete=,
// RUN: dpct --autocomplete==
// RUN: dpct --autocomplete=,,
// RUN: dpct --autocomplete=-
// RUN: dpct --autocomplete=##
// RUN: dpct --autocomplete=#

// RUN: dpct --autocomplete=--gen-build | FileCheck %s -check-prefix=DASHDASHGENBUILD
// DASHDASHGENBUILD: --gen-build-script
// RUN: dpct --autocomplete=-gen-build | FileCheck %s -check-prefix=DASHGENBUILD
// DASHGENBUILD: -gen-build-script
// RUN: dpct --autocomplete=foo | FileCheck %s -check-prefix=FOO
// FOO-NOT: foo
// RUN: dpct --autocomplete=--output-verbosity=#d | FileCheck %s -check-prefix=DASHDASHOUTPUTVEBD
// DASHDASHOUTPUTVEBD: detailed
// DASHDASHOUTPUTVEBD-NEXT: diagnostics
// RUN: dpct --autocomplete=-output-verbosity=#d | FileCheck %s -check-prefix=DASHOUTPUTVEBD
// DASHOUTPUTVEBD: detailed
// DASHOUTPUTVEBD-NEXT: diagnostics
// RUN: dpct --autocomplete=--output-verbosity= | FileCheck %s -check-prefix=DASHDASHOUTPUTVEBALL
// DASHDASHOUTPUTVEBALL: detailed
// DASHDASHOUTPUTVEBALL-NEXT: diagnostics
// DASHDASHOUTPUTVEBALL-NEXT: normal
// DASHDASHOUTPUTVEBALL-NEXT: silent
// RUN: dpct --autocomplete=-output-verbosity= | FileCheck %s -check-prefix=DASHOUTPUTVEBALL
// DASHOUTPUTVEBALL: detailed
// DASHOUTPUTVEBALL-NEXT: diagnostics
// DASHOUTPUTVEBALL-NEXT: normal
// DASHOUTPUTVEBALL-NEXT: silent

// RUN: dpct --autocomplete=foo#bar##--enable | FileCheck %s -check-prefix=ENABLECTAD
// ENABLECTAD: --enable-ctad
// RUN: dpct --autocomplete=foo#bar###--format-range=#a | FileCheck %s -check-prefix=FORMATRANGE-ALL
// FORMATRANGE-ALL: all

// RUN: dpct --autocomplete=--rule-file= | FileCheck %s -check-prefix=RULE_FILE_EQUAL
// RULE_FILE_EQUAL-NOT: --rule-file=
// RUN: dpct --autocomplete=--rule-file | FileCheck %s -check-prefix=RULE_FILE
// RULE_FILE: --rule-file
// RUN: dpct --autocomplete=-p= | FileCheck %s -check-prefix=DATA_BASE_EQUAL
// DATA_BASE_EQUAL-NOT: -p=
// RUN: dpct --autocomplete=-p | FileCheck %s -check-prefix=DATA_BASE
// DATA_BASE: -p

// RUN: dpct --autocomplete=--usm-level=#none,restricted#--use-explicit-namespace=#cl,sycl, | FileCheck %s -check-prefix=UEN_ALL
// UEN_ALL: cl,sycl,cl
// UEN_ALL-NEXT: cl,sycl,dpct
// UEN_ALL-NEXT: cl,sycl,none
// UEN_ALL-NEXT: cl,sycl,sycl
// UEN_ALL-NEXT: cl,sycl,sycl-math
// RUN: dpct --autocomplete=--usm-level=#none,restricted#--use-explicit-namespace=#cl,sycl,s | FileCheck %s -check-prefix=UEN_S
// UEN_S: cl,sycl,sycl
// UEN_S-NEXT: cl,sycl,sycl-math
