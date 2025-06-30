#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>

void fill_n_test() {
  // clang-format off
  // Start
  float *_de = NULL;
  float fill_value = 0.0;
  thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(static_cast<float *>(&_de[0]));
  thrust::fill_n(dev_ptr, 10, fill_value);
  // End
  // clang-format on
}
