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

void transform_exclusive_scan_test() {
  const int N = 6;
  int A[N] = {1, 0, 2, 2, 1, 3};
  thrust::host_vector<int> h_V(A, A + N);
  thrust::device_vector<int> d_V(A, A + N);
  thrust::negate<int> unary_op;
  thrust::plus<int> binary_op;
  // Start
  /*1*/ thrust::transform_exclusive_scan(thrust::host, h_V.begin(), h_V.end(),
                                         h_V.begin(), unary_op, 4, binary_op);
  /*2*/ thrust::transform_exclusive_scan(h_V.begin(), h_V.end(), h_V.begin(),
                                         unary_op, 4, binary_op);
  /*3*/ thrust::transform_exclusive_scan(thrust::device, d_V.begin(), d_V.end(),
                                         d_V.begin(), unary_op, 4, binary_op);
  /*4*/ thrust::transform_exclusive_scan(d_V.begin(), d_V.end(), d_V.begin(),
                                         unary_op, 4, binary_op);
  /*5*/ thrust::transform_exclusive_scan(thrust::host, A, A + N, A, unary_op, 4,
                                         binary_op);
  /*6*/ thrust::transform_exclusive_scan(A, A + N, A, unary_op, 4, binary_op);
  // End
}
