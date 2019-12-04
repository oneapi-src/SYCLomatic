// dpcpp async_exception.cpp -o run
// expect output:
// Caught asynchronous SYCL exception:
// Global size is not a multiple of local size -54 (CL_INVALID_WORK_GROUP_SIZE)
// EOE at file:/path/to/include/dpct/device.hpp, line:40

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>

int main(){
  dpct::get_default_queue().submit(
    [&](cl::sycl::handler &cgh) {
      cgh.parallel_for<class kernel>(
        cl::sycl::nd_range<3>(cl::sycl::range<3>(256, 1, 1), cl::sycl::range<3>(257, 1, 1)),
          [=](cl::sycl::nd_item<3> item_ct1) {
      });
  });
}
