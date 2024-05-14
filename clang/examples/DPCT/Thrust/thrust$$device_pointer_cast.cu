#include <algorithm>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/scatter.h>
#include <thrust/tuple.h>
// for cuda 12.0
#include <thrust/count.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

void device_pointer_cast_test(void) {
  int data[10];

  // Start
  thrust::device_ptr<int> begin = thrust::device_pointer_cast(&data[0]);
  // End
}
