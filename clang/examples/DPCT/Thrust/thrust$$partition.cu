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

struct is_even {
  __host__ __device__ bool operator()(const int &x) { return (x % 2) == 0; }
};
void is_partition_test() {
  int datas[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const int N = sizeof(datas) / sizeof(int);
  int stencil[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  thrust::host_vector<int> h_vdata(datas, datas + N);
  thrust::host_vector<int> h_vstencil(stencil, stencil + N);
  thrust::device_vector<int> d_v(datas, datas + N);
  thrust::host_vector<int> h_v(datas, datas + N);
  thrust::device_vector<int> d_vdata(datas, datas + N);
  thrust::device_vector<int> d_vstencil(stencil, stencil + N);

  // Start
  /*1*/ thrust::partition(thrust::host, h_v.begin(), h_v.end(), is_even());
  /*2*/ thrust::partition(h_v.begin(), h_v.end(), is_even());
  /*3*/ thrust::partition(thrust::host, h_vdata.begin(), h_vdata.end(),
                          h_vstencil.begin(), is_even());
  /*4*/ thrust::partition(h_vdata.begin(), h_vdata.end(), h_vstencil.begin(),
                          is_even());
  /*5*/ thrust::partition(thrust::device, d_v.begin(), d_v.end(), is_even());
  /*6*/ thrust::partition(d_v.begin(), d_v.end(), is_even());
  /*7*/ thrust::partition(thrust::device, d_vdata.begin(), d_vdata.end(),
                          d_vstencil.begin(), is_even());
  /*8*/ thrust::partition(d_vdata.begin(), d_vdata.end(), d_vstencil.begin(),
                          is_even());
  /*9*/ thrust::partition(thrust::host, datas, datas + N, is_even());
  /*10*/ thrust::partition(datas, datas + N, is_even());
  /*11*/ thrust::partition(thrust::host, datas, datas + N, stencil, is_even());
  /*12*/ thrust::partition(datas, datas + N, stencil, is_even());
  // End
}
