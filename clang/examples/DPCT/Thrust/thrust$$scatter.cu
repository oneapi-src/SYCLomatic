#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/replace.h>
#include <thrust/scatter.h>

void scatter() {
  // clang-format off
  // Start
  int values[10] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  thrust::host_vector<int> h_values(values, values + 10);
  int map[10] = {0, 5, 1, 6, 2, 7, 3, 8, 4, 9};
  thrust::host_vector<int> h_map(map, map + 10);
  thrust::host_vector<int> h_output(10);
  /*1*/ thrust::scatter(thrust::seq, h_values.begin(), h_values.end(),h_map.begin(), h_output.begin());
  /*2*/ thrust::scatter(h_values.begin(), h_values.end(), h_map.begin(),h_output.begin());
  // End
  // clang-format on
}
