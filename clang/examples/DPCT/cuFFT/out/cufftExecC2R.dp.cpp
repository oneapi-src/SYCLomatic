#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/fft_utils.hpp>

void test(dpct::fft::fft_engine_ptr plan, sycl::float2 *in, float *out) {
  // Start
  plan->compute<sycl::float2, float>(in, out,
                                     dpct::fft::fft_direction::backward);
  // End
}
