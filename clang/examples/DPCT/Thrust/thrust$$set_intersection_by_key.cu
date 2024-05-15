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

void set_intersection_by_key_test() {
  const int N = 6, M = 7, P = 3;
  int Akey[N] = {1, 3, 5, 7, 9, 11};
  int Avalue[N] = {0, 0, 0, 0, 0, 0};
  int Bkey[M] = {1, 1, 2, 3, 5, 8, 13};
  int Ckey[P];
  int Cvalue[P];

  thrust::host_vector<int> h_VAkey(Akey, Akey + N);
  thrust::host_vector<int> h_VAvalue(Avalue, Avalue + N);

  thrust::host_vector<int> h_VBkey(Bkey, Bkey + M);

  thrust::host_vector<int> h_VCkey(Ckey, Ckey + P);
  thrust::host_vector<int> h_VCvalue(Cvalue, Cvalue + P);
  typedef thrust::pair<thrust::host_vector<int>::iterator,
                       thrust::host_vector<int>::iterator>
      iter_pair;
  thrust::device_vector<int> d_VAkey(Akey, Akey + N);
  thrust::device_vector<int> d_VAvalue(Avalue, Avalue + N);
  thrust::device_vector<int> d_VBkey(Bkey, Bkey + M);
  thrust::device_vector<int> d_VCkey(Ckey, Ckey + P);
  thrust::device_vector<int> d_VCvalue(Cvalue, Cvalue + P);
  // Start
  /*1*/ thrust::set_intersection_by_key(
      thrust::host, h_VAkey.begin(), h_VAkey.end(), h_VBkey.begin(),
      h_VBkey.end(), h_VAvalue.begin(), h_VCkey.begin(), h_VCvalue.begin());
  /*2*/ thrust::set_intersection_by_key(
      h_VAkey.begin(), h_VAkey.end(), h_VBkey.begin(), h_VBkey.end(),
      h_VAvalue.begin(), h_VCkey.begin(), h_VCvalue.begin());
  /*3*/ thrust::set_intersection_by_key(
      thrust::host, h_VAkey.begin(), h_VAkey.end(), h_VBkey.begin(),
      h_VBkey.end(), h_VAvalue.begin(), h_VCkey.begin(), h_VCvalue.begin(),
      thrust::greater<int>());
  /*4*/ thrust::set_intersection_by_key(
      h_VAkey.begin(), h_VAkey.end(), h_VBkey.begin(), h_VBkey.end(),
      h_VAvalue.begin(), h_VCkey.begin(), h_VCvalue.begin(),
      thrust::greater<int>());
  /*5*/ thrust::set_intersection_by_key(
      thrust::device, d_VAkey.begin(), d_VAkey.end(), d_VBkey.begin(),
      d_VBkey.end(), d_VAvalue.begin(), d_VCkey.begin(), d_VCvalue.begin());
  /*6*/ thrust::set_intersection_by_key(
      d_VAkey.begin(), d_VAkey.end(), d_VBkey.begin(), d_VBkey.end(),
      d_VAvalue.begin(), d_VCkey.begin(), d_VCvalue.begin());
  /*7*/ thrust::set_intersection_by_key(
      thrust::device, d_VAkey.begin(), d_VAkey.end(), d_VBkey.begin(),
      d_VBkey.end(), d_VAvalue.begin(), d_VCkey.begin(), d_VCvalue.begin(),
      thrust::greater<int>());
  /*8*/ thrust::set_intersection_by_key(
      d_VAkey.begin(), d_VAkey.end(), d_VBkey.begin(), d_VBkey.end(),
      d_VAvalue.begin(), d_VCkey.begin(), d_VCvalue.begin(),
      thrust::greater<int>());
  /*9*/ thrust::set_intersection_by_key(thrust::host, Akey, Akey + N, Bkey,
                                        Bkey + M, Avalue, Ckey, Cvalue);
  /*10*/ thrust::set_intersection_by_key(Akey, Akey + N, Bkey, Bkey + M, Avalue,
                                         Ckey, Cvalue);
  /*11*/ thrust::set_intersection_by_key(thrust::host, Akey, Akey + N, Bkey,
                                         Bkey + M, Avalue, Ckey, Cvalue,
                                         thrust::greater<int>());
  /*12*/ thrust::set_intersection_by_key(Akey, Akey + N, Bkey, Bkey + M, Avalue,
                                         Ckey, Cvalue, thrust::greater<int>());
  // End
}
