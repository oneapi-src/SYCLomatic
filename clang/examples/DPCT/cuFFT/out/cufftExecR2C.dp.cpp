#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/fft_utils.hpp>

void test(dpct::fft::fft_engine_ptr plan, float *in, sycl::float2 *out) {
  // Start
  plan->compute<float, sycl::float2>(in, out,
                                     dpct::fft::fft_direction::forward);
  // End
}
