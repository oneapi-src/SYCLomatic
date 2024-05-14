#include <thrust/device_delete.h>
#include <thrust/device_malloc.h>
#include <thrust/device_new.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/memory.h>

void malloc_test() {
  const int N = 137;
  int val = 46;

  thrust::device_system_tag device_sys;

  // Start
  /*1*/ thrust::pointer<int, thrust::device_system_tag> ptr =
      thrust::malloc<int>(device_sys, N);
  /*2*/ thrust::pointer<void, thrust::device_system_tag> void_ptr =
      thrust::malloc(device_sys, N);
  // End
}
