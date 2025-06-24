#include <thrust/device_delete.h>
#include <thrust/device_malloc.h>
#include <thrust/device_new.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/memory.h>

void return_temporary_buffer_test() {

  // clang-format off
  // Start
  const int N = 100;
  typedef thrust::pair<thrust::pointer<int, thrust::device_system_tag>, std::ptrdiff_t> ptr_and_size_t;
  thrust::device_system_tag device_sys;
  ptr_and_size_t ptr_and_size = thrust::get_temporary_buffer<int>(device_sys, N);
  thrust::return_temporary_buffer(device_sys, ptr_and_size.first,ptr_and_size.second);
  // End
  // clang-format on
}
