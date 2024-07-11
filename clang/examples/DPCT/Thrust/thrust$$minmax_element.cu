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

struct compare_key_value {
  __host__ __device__ bool operator()(int lhs, int rhs) const {
    return lhs < rhs;
  }
};

void minmax_element_test() {
  const int N = 6;
  int data[N] = {1, 0, 2, 2, 1, 3};
  thrust::host_vector<int> h_values(data, data + N);
  thrust::device_vector<int> d_values(data, data + N);
  // Start
  /*1*/ thrust::minmax_element(thrust::host, h_values.begin(), h_values.end());
  /*2*/ thrust::minmax_element(h_values.begin(), h_values.end());
  /*3*/ thrust::minmax_element(thrust::host, h_values.begin(),
                               h_values.begin() + 4, compare_key_value());
  /*4*/ thrust::minmax_element(h_values.begin(), h_values.begin() + 4,
                               compare_key_value());
  /*5*/ thrust::minmax_element(thrust::device, d_values.begin(),
                               d_values.end());
  /*6*/ thrust::minmax_element(d_values.begin(), d_values.end());
  /*7*/ thrust::minmax_element(thrust::device, d_values.begin(), d_values.end(),
                               compare_key_value());
  /*8*/ thrust::minmax_element(d_values.begin(), d_values.end(),
                               compare_key_value());
  /*9*/ thrust::minmax_element(thrust::host, data, data + N);
  /*10*/ thrust::minmax_element(data, data + N);
  /*11*/ thrust::minmax_element(thrust::host, data, data + N,
                                compare_key_value());
  /*12*/ thrust::minmax_element(data, data + N, compare_key_value());
  // End
}
