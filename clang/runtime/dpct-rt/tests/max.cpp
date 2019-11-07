// This test case is target to run at Windows platform:
// Previous there is issue when compile migrated code for there is MACRO version of max() in
// windows.
// RUN: dpcpp max.cpp -o max -I/path/to/sycl/include -I/path/to/dpct/include
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

int foo(int i) {
  return cl::sycl::max(0, i);
}

int main() {
  return foo(23);
}

