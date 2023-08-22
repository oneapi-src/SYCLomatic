#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/fft_utils.hpp>

void test(dpct::fft::fft_engine_ptr *plan, int nx, dpct::fft::fft_type type,
          int num_of_trans) {
  // Start
  *plan = dpct::fft::fft_engine::create(&dpct::get_default_queue(), nx, type,
                                        num_of_trans);
  // End
}
