#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/fft_utils.hpp>

void test(dpct::fft::fft_engine_ptr plan, double *in, sycl::double2 *out) {
  // Start
  plan->compute<double, sycl::double2>(in, out,
                                       dpct::fft::fft_direction::forward);
  // End
}
