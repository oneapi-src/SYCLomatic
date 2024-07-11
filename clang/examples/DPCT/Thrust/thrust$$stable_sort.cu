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

void stable_sort_test() {
  const int N = 6;
  int datas[N] = {1, 4, 2, 8, 5, 7};
  thrust::host_vector<int> h_v(datas, datas + N);
  thrust::device_vector<int> d_v(datas, datas + N);

  // Start
  /*1*/ thrust::stable_sort(thrust::host, h_v.begin(), h_v.end());
  /*2*/ thrust::stable_sort(h_v.begin(), h_v.end());
  /*3*/ thrust::stable_sort(thrust::host, h_v.begin(), h_v.end(),
                            thrust::greater<int>());
  /*4*/ thrust::stable_sort(h_v.begin(), h_v.end(), thrust::greater<int>());
  /*5*/ thrust::stable_sort(thrust::device, d_v.begin(), d_v.end());
  /*6*/ thrust::stable_sort(d_v.begin(), d_v.end());
  /*7*/ thrust::stable_sort(thrust::device, d_v.begin(), d_v.end(),
                            thrust::greater<int>());
  /*8*/ thrust::stable_sort(d_v.begin(), d_v.end(), thrust::greater<int>());
  /*9*/ thrust::stable_sort(thrust::host, datas, datas + N);
  /*10*/ thrust::stable_sort(datas, datas + N);
  /*11*/ thrust::stable_sort(thrust::host, datas, datas + N,
                             thrust::greater<int>());
  /*12*/ thrust::stable_sort(datas, datas + N, thrust::greater<int>());
  // End
}