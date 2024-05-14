#include <thrust/device_delete.h>
#include <thrust/device_malloc.h>
#include <thrust/device_new.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/memory.h>

void device_new_test() {
  const int N = 137;
  int val = 46;

  thrust::device_ptr<int> d_mem = thrust::device_malloc<int>(N);

  // Start
  /*1*/ thrust::device_ptr<int> d_array1 = thrust::device_new<int>(d_mem, N);
  /*2*/ thrust::device_ptr<int> d_array2 =
      thrust::device_new<int>(d_mem, val, N);
  /*3*/ thrust::device_ptr<int> d_array3 = thrust::device_new<int>(N);
  // End
}
