#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>

// This file is compiled together with test_default_queue_2.cpp.
cl::sycl::queue &get_queue_1(void) { return dpct::get_default_queue(); }
