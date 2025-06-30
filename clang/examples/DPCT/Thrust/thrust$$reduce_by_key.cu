#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

void reduce_by_key_test() {
  // clang-format off
  // Start
  thrust::host_vector<int> h;
  thrust::device_vector<int> d;
  thrust::not_equal_to<int> bp;
  thrust::multiplies<int> bo1;
  /*1*/ thrust::reduce_by_key(thrust::host, h.begin(), h.end(), h.begin(), h.end(), h.begin(), bp, bo1);
  /*2*/thrust::reduce_by_key(thrust::device, d.begin(), d.end(), d.begin(), d.end(), d.begin(), bp);
  /*3*/thrust::reduce_by_key(d.begin(), d.end(), d.begin(), d.end(), d.begin(), bp, bo1);
  /*4*/thrust::reduce_by_key(thrust::host, h.begin(), h.end(), thrust::constant_iterator<int>(1), h.end(), h.begin());
  /*5*/thrust::reduce_by_key(d.begin(), d.end(), d.begin(), d.end(), d.begin(), bp);
  /*6*/thrust::reduce_by_key(d.begin(), d.end(), d.begin(), d.end(), d.begin());
  // End
  // clang-format on
}
