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

void remove_copy_test() {
  const int N = 6;
  int A[N] = {-2, 0, -1, 0, 1, 2};
  int B[N - 2];
  int result[N - 2];
  int V[N] = {-2, 0, -1, 0, 1, 2};

  thrust::host_vector<int> h_V(A, A + N);
  thrust::host_vector<int> h_result(B, B + N - 2);
  thrust::device_vector<int> d_V(A, A + N);
  thrust::device_vector<int> d_result(B, B + N - 2);

  // Start
  /*1*/ thrust::remove_copy(thrust::host, h_V.begin(), h_V.end(),
                            h_result.begin(), 0);
  /*2*/ thrust::remove_copy(h_V.begin(), h_V.end(), h_result.begin(), 0);
  /*3*/ thrust::remove_copy(thrust::device, d_V.begin(), d_V.end(),
                            d_result.begin(), 0);
  /*4*/ thrust::remove_copy(d_V.begin(), d_V.end(), d_result.begin(), 0);
  /*5*/ thrust::remove_copy(thrust::host, V, V + N, result, 0);
  /*6*/ thrust::remove_copy(V, V + N, result, 0);
  // End
}
