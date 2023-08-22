#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/fft_utils.hpp>

void test(dpct::fft::fft_engine_ptr plan) {
  // Start
  dpct::fft::fft_engine::destroy(plan);
  // End
}
