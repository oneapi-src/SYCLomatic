#include <vector>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/find.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>
#include <thrust/mismatch.h>
#include <thrust/partition.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/replace.h>
#include <thrust/reverse.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/transform_scan.h>
#include <thrust/unique.h>

void partition_point() {
  int data[] = {2, 4, 6, 8, 10, 1, 3, 5, 7, 9};
  thrust::host_vector<int> h_v(data, data + 10);
  thrust::device_vector<int> d_v(data, data + 10);

  auto up = [](int x) -> bool { return x < 23; };
  // Start
  /*1*/ thrust::partition_point(thrust::seq, h_v.begin(), h_v.end(), up);
  /*2*/ thrust::partition_point(thrust::device, d_v.begin(), d_v.end(), up);
  /*3*/ thrust::partition_point(thrust::host, data, data + 10, up);
  /*4*/ thrust::partition_point(h_v.begin(), h_v.end(), up);
  /*5*/ thrust::partition_point(d_v.begin(), d_v.end(), up);
  /*6*/ thrust::partition_point(data, data + 10, up);
  // End
}
