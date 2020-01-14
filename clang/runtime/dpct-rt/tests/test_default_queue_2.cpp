// This test case is used to test if only one default queue has been created.
// dpcpp test_default_queue_1.cpp test_default_queue_2.cpp -o test_default_queue
// ./test_default_queue
// Test Passed

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>

extern cl::sycl::queue &get_queue_1(void);

cl::sycl::queue &get_queue_2(void) { return dpct::get_default_queue(); }

int main() {
  if (&get_queue_1() == &get_queue_2()) {
    std::cout << "Test Passed\n"
              << "\n";
    return 0;
  } else {
    std::cout << "Test Failed\n"
              << "\n";
    return -1;
  }
}
