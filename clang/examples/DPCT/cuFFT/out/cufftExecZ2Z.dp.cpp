#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/fft_utils.hpp>

void test(dpct::fft::fft_engine_ptr plan, sycl::double2 *in, sycl::double2 *out,
          int dir) {
  // Start
  plan->compute<sycl::double2, sycl::double2>(
      in, out,
      dir == 1 ? dpct::fft::fft_direction::backward
               : dpct::fft::fft_direction::forward);
  // End
}
