// UNSUPPORTED: system-windows
// intercept-build only supports Linux
//
// ------ prepare test directory
// RUN: cd %T
// RUN: cp %s foo.cu
// RUN: cp %S/Makefile ./Makefile
// RUN: cp %S/CMakeLists.txt ./CMakeLists.txt
// RUN: cp %S/bar.cpp bar.cpp
// RUN: intercept-build make > intercept_log.txt 2>&1
// RUN: grep "Warning: cmake is called during make running: please run make first" intercept_log.txt
#include <iostream>

int main() {
  return 0;
}
