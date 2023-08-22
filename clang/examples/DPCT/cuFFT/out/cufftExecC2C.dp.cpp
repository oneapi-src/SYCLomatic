#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/fft_utils.hpp>

void test(dpct::fft::fft_engine_ptr plan, sycl::float2 *in, sycl::float2 *out,
          int dir) {
  // Start
  plan->compute<sycl::float2, sycl::float2>(
      in, out,
      dir == 1 ? dpct::fft::fft_direction::backward
               : dpct::fft::fft_direction::forward);
  // End
}
