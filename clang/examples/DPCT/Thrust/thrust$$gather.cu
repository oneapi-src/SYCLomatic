#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <algorithm>
#include <thrust/inner_product.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/tuple.h>
#include <thrust/device_ptr.h>
#include <thrust/count.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

void gather_test() {
  // clang-format off
  // Start
  int values[10] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  thrust::host_vector<int> h_values(values, values + 10);
  int map[10] = {0, 2, 4, 6, 8, 1, 3, 5, 7, 9};
  thrust::host_vector<int> h_map(map, map + 10);
  thrust::host_vector<int> h_output(10);
  thrust::device_vector<int> d_values(values, values + 10);
  thrust::device_vector<int> d_map(map, map + 10);
  thrust::device_vector<int> d_output(10);
  /*1*/ thrust::gather(d_map.begin(), d_map.end(), d_values.begin(),d_output.begin());
  /*2*/ thrust::gather(thrust::device, d_map.begin(), d_map.end(),d_values.begin(), d_output.begin());
  /*3*/ thrust::gather(thrust::seq, h_map.begin(), h_map.end(),h_values.begin(), h_output.begin());
  /*4*/ thrust::gather(h_map.begin(), h_map.end(), h_values.begin(),h_output.begin());
  // End
  // clang-format on
}
