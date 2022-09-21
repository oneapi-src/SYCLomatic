// UNSUPPORTED: -windows-
// intercept-build not found on windows
//
// ------ prepare test directory
// RUN: cd %T
// RUN: rm -rf intercept-build
// RUN: mkdir  intercept-build
// RUN: cd     intercept-build
// RUN: cp %s intercept.c
//
// ------ create makefile
// RUN: echo "libintercept.a     :       intercept.c          "          >  intercept-Makefile
// RUN: echo "	%clangxx -c -fpic intercept.c                 "          >> intercept-Makefile
// RUN: echo "	%clangxx -shared -o libintercept.a intercept.o"          >> intercept-Makefile
//
// ------ test intercept-build
// RUN: intercept-build --cdb intercept.json make -f intercept-Makefile
//
// ------ ensure that '-shared' option is recorded in compilation database
// RUN: FileCheck --input-file intercept.json %s
//
// ------ cleanup test directory
// RUN: cd ..
// RUN: rm -rf ./intercept-build

// CHECK: "command": "cc -c --driver-mode=g++ -fpic intercept.c",
// CHECK: -shared -o libintercept.a

void foo()
{
}
