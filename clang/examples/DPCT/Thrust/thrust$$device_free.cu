#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

void thrust_device_free() {
  // clang-format off
  // Start
  thrust::device_ptr<thrust::complex<double>> d_ptr = thrust::device_malloc<thrust::complex<double>>(1);
  thrust::device_free(d_ptr);
  // End
  // clang-format on
}
