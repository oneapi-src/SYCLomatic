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

void is_sorted_test() {
  const int N = 6;
  int datas[N] = {1, 4, 2, 8, 5, 7};

  thrust::host_vector<int> h_v(datas, datas + N);
  thrust::device_vector<int> d_v(datas, datas + N);
  thrust::greater<int> comp;

  // Start
  /*1*/ thrust::is_sorted(thrust::host, h_v.begin(), h_v.end());
  /*2*/ thrust::is_sorted(h_v.begin(), h_v.end());
  /*3*/ thrust::is_sorted(thrust::host, h_v.begin(), h_v.end(), comp);
  /*4*/ thrust::is_sorted(h_v.begin(), h_v.end(), comp);
  /*5*/ thrust::is_sorted(h_v.begin(), h_v.end(), comp);
  /*6*/ thrust::is_sorted(thrust::device, d_v.begin(), d_v.end());
  /*7*/ thrust::is_sorted(thrust::device, d_v.begin(), d_v.end(), comp);
  /*8*/ thrust::is_sorted(d_v.begin(), d_v.end(), comp);
  /*9*/ thrust::is_sorted(thrust::host, datas, datas + N);
  /*10*/ thrust::is_sorted(datas, datas + N);
  /*11*/ thrust::is_sorted(thrust::host, datas, datas + N, comp);
  /*12*/ thrust::is_sorted(datas, datas + N, comp);
  // End
}
