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

void stable_sort_by_key() {
  // clang-format off
  // Start
  const int N = 7;
  int A[N] = {1, 3, 3, 3, 2, 2, 1}; // input keys
  int B[N] = {9, 8, 7, 6, 5, 4, 3}; // input values
  thrust::device_vector<int> d_keys(A, A + N);
  thrust::device_vector<int> d_values(B, B + N);
  thrust::device_vector<int> d_output_keys(N);
  thrust::device_vector<int> d_output_values(N);
  thrust::equal_to<int> binary_pred;
  typedef thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> iter_pair;
  thrust::device_vector<iter_pair> new_last_vec(1);
  /*1*/ *new_last_vec.begin() = thrust::unique_by_key_copy(thrust::device, d_keys.begin(), d_keys.end(), d_values.begin(), d_output_keys.begin(), d_output_values.begin(), binary_pred);
  /*2*/ *new_last_vec.begin() = thrust::unique_by_key_copy(d_keys.begin(), d_keys.end(), d_values.begin(), d_output_keys.begin(), d_output_values.begin(), binary_pred);
  /*3*/ *new_last_vec.begin() = thrust::unique_by_key_copy(d_keys.begin(), d_keys.end(), d_values.begin(), d_output_keys.begin(), d_output_values.begin());
  /*4*/ *new_last_vec.begin() = thrust::unique_by_key_copy(thrust::device, d_keys.begin(), d_keys.end(), d_values.begin(), d_output_keys.begin(), d_output_values.begin());
  // End
  // clang-format on
}
