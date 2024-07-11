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

// CHECK: #include <oneapi/dpl/memory>
#include <thrust/equal.h>
#include <thrust/uninitialized_copy.h>

// for cuda 12.0
#include <thrust/iterator/constant_iterator.h>
#include <thrust/partition.h>
#include <thrust/scatter.h>

void unique_by_key_test() {
  thrust::host_vector<int> h_keys, h_values;
  thrust::device_vector<int> d_keys, d_values;
  thrust::equal_to<int> binary_pred;
  const int N = 7;
  int A[N]; // keys
  int B[N]; // values

  // Start
  /*1*/ thrust::unique_by_key(thrust::host, h_keys.begin(), h_keys.end(),
                              h_values.begin());
  /*2*/ thrust::unique_by_key(h_keys.begin(), h_keys.end(), h_values.begin());
  /*3*/ thrust::unique_by_key(thrust::host, h_keys.begin(), h_keys.end(),
                              h_values.begin(), binary_pred);
  /*4*/ thrust::unique_by_key(h_keys.begin(), h_keys.end(), h_values.begin(),
                              binary_pred);
  /*5*/ thrust::unique_by_key(thrust::device, d_keys.begin(), d_keys.end(),
                              d_values.begin());
  /*6*/ thrust::unique_by_key(d_keys.begin(), d_keys.end(), d_values.begin());
  /*7*/ thrust::unique_by_key(thrust::device, d_keys.begin(), d_keys.end(),
                              d_values.begin(), binary_pred);
  /*8*/ thrust::unique_by_key(d_keys.begin(), d_keys.end(), d_values.begin(),
                              binary_pred);
  /*9*/ thrust::unique_by_key(thrust::host, A, A + N, B);
  /*10*/ thrust::unique_by_key(A, A + N, B);
  /*11*/ thrust::unique_by_key(thrust::host, A, A + N, B, binary_pred);
  /*12*/ thrust::unique_by_key(A, A + N, B, binary_pred);
  // End
}
