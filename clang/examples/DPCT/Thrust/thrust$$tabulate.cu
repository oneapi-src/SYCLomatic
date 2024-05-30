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

void tabulate_test() {
  const int N = 10;
  int A[N];
  thrust::host_vector<int> h_V(A, A + N);
  thrust::device_vector<int> d_V(A, A + N);

  // Start
  /*1*/ thrust::tabulate(thrust::host, h_V.begin(), h_V.end(),
                         thrust::negate<int>());
  /*2*/ thrust::tabulate(h_V.begin(), h_V.end(), thrust::negate<int>());
  /*3*/ thrust::tabulate(thrust::device, d_V.begin(), d_V.end(),
                         thrust::negate<int>());
  /*4*/ thrust::tabulate(d_V.begin(), d_V.end(), thrust::negate<int>());
  /*5*/ thrust::tabulate(thrust::host, A, A + N, thrust::negate<int>());
  /*6*/ thrust::tabulate(A, A + N, thrust::negate<int>());
  // End
}
