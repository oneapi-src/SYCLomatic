#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/fft_utils.hpp>

void test(dpct::fft::fft_engine_ptr *plan, int nx, int ny,
          dpct::fft::fft_type type) {
  // Start
  *plan =
      dpct::fft::fft_engine::create(&dpct::get_default_queue(), nx, ny, type);
  // End
}
