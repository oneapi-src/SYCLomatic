#include <thrust/device_delete.h>
#include <thrust/device_malloc.h>
#include <thrust/device_new.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/memory.h>

void thrust_free() {
  // clang-format off
  // Start
  const int N = 100;
  thrust::device_system_tag device_sys;
  thrust::pointer<int, thrust::device_system_tag> ptr = thrust::malloc<int>(device_sys, N);
  thrust::free(device_sys, ptr);
  // End
  // clang-format on
}
