#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/fft_utils.hpp>

void test(dpct::fft::fft_engine_ptr plan, sycl::double2 *in, double *out) {
  // Start
  plan->compute<sycl::double2, double>(in, out,
                                       dpct::fft::fft_direction::backward);
  // End
}
