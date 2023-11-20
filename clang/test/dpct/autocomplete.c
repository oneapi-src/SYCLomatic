// UNSUPPORTED: system-windows

// RUN: dpct --autocomplete=
// RUN: dpct --autocomplete=,
// RUN: dpct --autocomplete==
// RUN: dpct --autocomplete=,,
// RUN: dpct --autocomplete=- | FileCheck %s -check-prefix=DASH
// Notice: When modify DASH prefix check, need modify behavior_tests/src/bt-autocomplete/do_test.py
// in SYCLomatic-test repo too.
// DASH: --always-use-async-handler
// DASH-NEXT: --analysis-scope-path
// DASH-NEXT: --assume-nd-range-dim=
// DASH-NEXT: --build-script-file
// DASH-NEXT: --change-cuda-files-extension-only
// DASH-NEXT: --check-unicode-security
// DASH-NEXT: --comments
// DASH-NEXT: --compilation-database
// DASH-NEXT: --cuda-include-path
// DASH-NEXT: --enable-ctad
// DASH-NEXT: --enable-profiling
// DASH-NEXT: --extra-arg
// DASH-NEXT: --format-range=
// DASH-NEXT: --format-style=
// DASH-NEXT: --gen-build-script
// DASH-NEXT: --gen-helper-function
// DASH-NEXT: --help
// DASH-NEXT: --helper-function-dir
// DASH-NEXT: --helper-function-preference=
// DASH-NEXT: --in-root
// DASH-NEXT: --in-root-exclude
// DASH-NEXT: --intercept-build
// DASH-NEXT: --keep-original-code
// DASH-NEXT: --no-cl-namespace-inline
// DASH-NEXT: --no-dpcpp-extensions=
// DASH-NEXT: --no-dry-pattern
// DASH-NEXT: --no-incremental-migration
// DASH-NEXT: --optimize-migration
// DASH-NEXT: --out-root
// DASH-NEXT: --output-file
// DASH-NEXT: --output-verbosity=
// DASH-NEXT: --process-all
// DASH-NEXT: --query-api-mapping
// DASH-NEXT: --report-file-prefix
// DASH-NEXT: --report-format=
// DASH-NEXT: --report-only
// DASH-NEXT: --report-type=
// DASH-NEXT: --rule-file
// DASH-NEXT: --stop-on-parse-err
// DASH-NEXT: --suppress-warnings
// DASH-NEXT: --suppress-warnings-all
// DASH-NEXT: --sycl-named-lambda
// DASH-NEXT: --use-dpcpp-extensions=
// DASH-NEXT: --use-experimental-features=
// DASH-NEXT: --use-explicit-namespace=
// DASH-NEXT: --usm-level=
// DASH-NEXT: --version
// DASH-NEXT: -p
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
