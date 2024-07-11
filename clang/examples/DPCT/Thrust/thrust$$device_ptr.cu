#include <thrust/device_delete.h>
#include <thrust/device_malloc.h>
#include <thrust/device_new.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/memory.h>

void device_ptr_test() {
  const int N = 137;
  int val = 46;

  // Start
  thrust::device_ptr<int> d_mem = thrust::device_malloc<int>(N);
  // End
}
