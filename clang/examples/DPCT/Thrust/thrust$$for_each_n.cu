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

struct add_functor {
  __host__ __device__ void operator()(int &x) { x++; }
};
void for_each_n_test() {
  const int N = 3;
  int A[N] = {0, 1, 2};
  thrust::host_vector<int> h_V(A, A + N);
  thrust::device_vector<int> d_V(A, A + N);

  // Start
  /*1*/ thrust::for_each_n(thrust::host, h_V.begin(), h_V.size(),
                           add_functor());
  /*2*/ thrust::for_each_n(h_V.begin(), h_V.size(), add_functor());
  /*3*/ thrust::for_each_n(thrust::device, d_V.begin(), d_V.size(),
                           add_functor());
  /*4*/ thrust::for_each_n(d_V.begin(), d_V.size(), add_functor());
  /*5*/ thrust::for_each_n(thrust::host, A, N, add_functor());
  /*6*/ thrust::for_each_n(A, N, add_functor());
  // End
}
