#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <thrust/adjacent_difference.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>

void inner_product_test() {
  // clang-format off
  // Start
  thrust::multiplies<int> bo1;
  thrust::multiplies<int> bo2;
  thrust::host_vector<int> h;
  thrust::device_vector<int> d;
  /*1*/ thrust::inner_product(thrust::host, h.begin(), h.end(), h.begin(), 1);
  /*2*/ thrust::inner_product(thrust::device, d.begin(), d.end(), d.begin(), 1, bo1, bo2);
  /*3*/ thrust::inner_product(d.begin(), d.end(), d.begin(), 1);
  /*4*/ thrust::inner_product(d.begin(), d.end(), d.begin(), 1, bo1, bo2);
  // End
  // clang-format on
}
