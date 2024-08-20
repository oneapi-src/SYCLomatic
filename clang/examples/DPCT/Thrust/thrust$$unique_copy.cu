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

void unique_copy_test() {
  const int N = 7;
  int A[N] = {1, 3, 3, 3, 2, 2, 1};
  int B[N];
  const int M = N - 3;
  thrust::host_vector<int> h_V(A, A + N);
  thrust::device_vector<int> d_V(A, A + N);
  thrust::host_vector<int> h_result(B, B + M);
  thrust::device_vector<int> d_result(B, B + M);

  // Start
  /*1*/ thrust::unique_copy(thrust::host, h_V.begin(), h_V.end(),
                            h_result.begin());
  /*2*/ thrust::unique_copy(h_V.begin(), h_V.end(), h_result.begin());
  /*3*/ thrust::unique_copy(thrust::host, h_V.begin(), h_V.end(),
                            h_result.begin(), thrust::equal_to<int>());
  /*4*/ thrust::unique_copy(h_V.begin(), h_V.end(), h_result.begin(),
                            thrust::equal_to<int>());
  /*5*/ thrust::unique_copy(thrust::device, d_V.begin(), d_V.end(),
                            d_result.begin());
  /*6*/ thrust::unique_copy(d_V.begin(), d_V.end(), d_result.begin());
  /*7*/ thrust::unique_copy(thrust::device, d_V.begin(), d_V.end(),
                            d_result.begin(), thrust::equal_to<int>());
  /*8*/ thrust::unique_copy(d_V.begin(), d_V.end(), d_result.begin(),
                            thrust::equal_to<int>());
  /*9*/ thrust::unique_copy(thrust::host, A, A + N, B);
  /*10*/ thrust::unique_copy(A, A + N, B);
  /*11*/ thrust::unique_copy(thrust::host, A, A + N, B,
                             thrust::equal_to<int>());
  /*12*/ thrust::unique_copy(A, A + N, B, thrust::equal_to<int>());
  // End
}
