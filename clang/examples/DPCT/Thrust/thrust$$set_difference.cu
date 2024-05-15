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

void set_difference_test() {
  const int N = 7, M = 5, P = 3;
  int A[N] = {0, 1, 3, 4, 5, 6, 9};
  int B[M] = {1, 3, 5, 7, 9};
  int C[P];
  thrust::host_vector<int> h_VA(A, A + N);
  thrust::host_vector<int> h_VB(B, B + M);
  thrust::host_vector<int> h_VC(C, C + P);
  thrust::device_vector<int> d_VA(A, A + N);
  thrust::device_vector<int> d_VB(B, B + M);
  thrust::device_vector<int> d_VC(C, C + P);

  // Start
  /*1*/ thrust::set_difference(thrust::host, h_VA.begin(), h_VA.end(),
                               h_VB.begin(), h_VB.end(), h_VC.begin());
  /*2*/ thrust::set_difference(h_VA.begin(), h_VA.end(), h_VB.begin(),
                               h_VB.end(), h_VC.begin());
  /*3*/ thrust::set_difference(thrust::host, h_VA.begin(), h_VA.end(),
                               h_VB.begin(), h_VB.end(), h_VC.begin(),
                               thrust::greater<int>());
  /*4*/ thrust::set_difference(h_VA.begin(), h_VA.end(), h_VB.begin(),
                               h_VB.end(), h_VC.begin(),
                               thrust::greater<int>());
  /*5*/ thrust::set_difference(thrust::device, d_VA.begin(), d_VA.end(),
                               d_VB.begin(), d_VB.end(), d_VC.begin());
  /*6*/ thrust::set_difference(d_VA.begin(), d_VA.end(), d_VB.begin(),
                               d_VB.end(), d_VC.begin());
  /*7*/ thrust::set_difference(thrust::device, d_VA.begin(), d_VA.end(),
                               d_VB.begin(), d_VB.end(), d_VC.begin(),
                               thrust::greater<int>());
  /*8*/ thrust::set_difference(d_VA.begin(), d_VA.end(), d_VB.begin(),
                               d_VB.end(), d_VC.begin(),
                               thrust::greater<int>());
  /*9*/ thrust::set_difference(thrust::host, A, A + N, B, B + M, C);
  /*10*/ thrust::set_difference(A, A + N, B, B + M, C);
  /*11*/ thrust::set_difference(thrust::host, A, A + N, B, B + M, C,
                                thrust::greater<int>());
  /*12*/ thrust::set_difference(A, A + N, B, B + M, C, thrust::greater<int>());
  // End
}
