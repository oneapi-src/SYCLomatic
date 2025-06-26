#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>
#include <thrust/scatter.h>
#include <thrust/tuple.h>

void exclusive_scan_test() {
  // clang-format off
  // Start
  std::vector<int> v, v2, v3, v4;
  thrust::maximum<int> binary_op;
  thrust::device_vector<int> tv, tv2, tv3, tv4;
  /*1*/ thrust::exclusive_scan(thrust::device, tv.begin(), tv.end(), tv2.begin());
  /*2*/ thrust::exclusive_scan(tv.begin(), tv.end(), tv2.begin());
  /*3*/ thrust::exclusive_scan(thrust::device, tv.begin(), tv.end(), tv2.begin(), 4);
  /*4*/ thrust::exclusive_scan(tv.begin(), tv.end(), tv2.begin(), 4);
  /*5*/ thrust::exclusive_scan(thrust::device, tv.begin(), tv.end(), tv2.begin(), 1, binary_op);
  /*6*/ thrust::exclusive_scan(tv.begin(), tv.end(), tv2.begin(), 1, binary_op);
  // End
  // clang-format on
}
