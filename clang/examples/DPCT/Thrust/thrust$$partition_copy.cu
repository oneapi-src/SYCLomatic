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

void partition_copy_test() {
  int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int S[] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  int result[10];
  const int N = sizeof(data) / sizeof(int);
  int *evens = result;
  int *odds = result + N / 2;

  thrust::host_vector<int> host_a(data, data + N);
  thrust::host_vector<int> host_evens(N / 2);
  thrust::host_vector<int> host_odds(N / 2);

  thrust::device_vector<int> device_a(data, data + N);
  thrust::device_vector<int> device_evens(N / 2);
  thrust::device_vector<int> device_odds(N / 2);
  thrust::host_vector<int> host_S(S, S + N);
  thrust::device_vector<int> device_S(S, S + N);

  // Start
  /*1*/ thrust::partition_copy(thrust::host, data, data + N, evens, odds,
                               is_even());
  /*2*/ thrust::partition_copy(thrust::host, host_a.begin(), host_a.begin() + N,
                               host_evens.begin(), host_odds.begin(),
                               is_even());
  /*3*/ thrust::partition_copy(thrust::device, device_a.begin(),
                               device_a.begin() + N, device_evens.begin(),
                               device_odds.begin(), is_even());
  /*4*/ thrust::partition_copy(data, data + N, evens, odds, is_even());
  /*5*/ thrust::partition_copy(host_a.begin(), host_a.begin() + N,
                               host_evens.begin(), host_odds.begin(),
                               is_even());
  /*6*/ thrust::partition_copy(device_a.begin(), device_a.begin() + N,
                               device_evens.begin(), device_odds.begin(),
                               is_even());
  /*7*/ thrust::partition_copy(thrust::host, data, data + N, S, evens, odds,
                               is_even());
  /*8*/ thrust::partition_copy(thrust::host, host_a.begin(), host_a.begin() + N,
                               host_S.begin(), host_evens.begin(),
                               host_odds.begin(), is_even());
  /*9*/ thrust::partition_copy(
      thrust::device, device_a.begin(), device_a.begin() + N, device_S.begin(),
      device_evens.begin(), device_odds.begin(), is_even());
  /*10*/ thrust::partition_copy(data, data + N, S, evens, odds, is_even());
  /*11*/ thrust::partition_copy(host_a.begin(), host_a.begin() + N,
                                host_S.begin(), host_evens.begin(),
                                host_odds.begin(), is_even());
  /*12*/ thrust::partition_copy(device_a.begin(), device_a.begin() + N,
                                device_S.begin(), device_evens.begin(),
                                device_odds.begin(), is_even());
  // End
}
