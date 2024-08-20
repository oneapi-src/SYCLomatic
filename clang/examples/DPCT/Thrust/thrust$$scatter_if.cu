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

struct is_even_scatter_if {
  __host__ __device__ bool operator()(int x) const { return (x % 2) == 0; }
};

void scatter_if() {

  const int N = 8;

  int V[N] = {10, 20, 30, 40, 50, 60, 70, 80};
  int M[N] = {0, 5, 1, 6, 2, 7, 3, 4};
  int S[N] = {1, 0, 1, 0, 1, 0, 1, 0};
  int D[N] = {0, 0, 0, 0, 0, 0, 0, 0};
  is_even_scatter_if pred;
  thrust::device_vector<int> d_V(V, V + N);
  thrust::device_vector<int> d_M(M, M + N);
  thrust::device_vector<int> d_S(S, S + N);
  thrust::device_vector<int> d_D(N);
  thrust::host_vector<int> h_V(V, V + N);
  thrust::host_vector<int> h_M(M, M + N);
  thrust::host_vector<int> h_S(S, S + N);
  thrust::host_vector<int> h_D(N);
  // Start
  /*1*/ thrust::scatter_if(thrust::host, V, V + 8, M, S, D);
  /*2*/ thrust::scatter_if(V, V + 8, M, S, D);
  /*3*/ thrust::scatter_if(thrust::host, V, V + 8, M, S, D, pred);
  /*4*/ thrust::scatter_if(V, V + 8, M, S, D, pred);
  /*5*/ thrust::scatter_if(thrust::device, d_V.begin(), d_V.end(), d_M.begin(),
                           d_S.begin(), d_D.begin());
  /*6*/ thrust::scatter_if(d_V.begin(), d_V.end(), d_M.begin(), d_S.begin(),
                           d_D.begin());
  /*7*/ thrust::scatter_if(thrust::device, d_V.begin(), d_V.end(), d_M.begin(),
                           d_S.begin(), d_D.begin(), pred);
  /*8*/ thrust::scatter_if(d_V.begin(), d_V.end(), d_M.begin(), d_S.begin(),
                           d_D.begin(), pred);
  /*9*/ thrust::scatter_if(thrust::host, h_V.begin(), h_V.end(), h_M.begin(),
                           h_S.begin(), h_D.begin());
  /*10*/ thrust::scatter_if(h_V.begin(), h_V.end(), h_M.begin(), h_S.begin(),
                            h_D.begin());
  /*11*/ thrust::scatter_if(thrust::host, h_V.begin(), h_V.end(), h_M.begin(),
                            h_S.begin(), h_D.begin(), pred);
  /*12*/ thrust::scatter_if(h_V.begin(), h_V.end(), h_M.begin(), h_S.begin(),
                            h_D.begin(), pred);
  // End
}
