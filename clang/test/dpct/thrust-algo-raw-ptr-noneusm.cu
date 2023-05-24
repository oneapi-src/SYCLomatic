// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --usm-level=none -out-root %T/thrust-algo-raw-ptr-noneusm %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17 -fsized-deallocation
// RUN: FileCheck --input-file %T/thrust-algo-raw-ptr-noneusm/thrust-algo-raw-ptr-noneusm.dp.cpp --match-full-lines %s


#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/unique.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/transform_scan.h>
#include <thrust/set_operations.h>
#include <thrust/tabulate.h>
#include <thrust/functional.h>
#include <thrust/remove.h>
#include <thrust/find.h>
#include <thrust/mismatch.h>
#include <thrust/replace.h>
#include <thrust/reverse.h>
#include <thrust/binary_search.h>

// for cuda 12.0
#include <thrust/iterator/constant_iterator.h>
#include <thrust/partition.h>

struct key_value
	{
		int key;
		int value;
		__host__ __device__
		bool operator!=( struct key_value &tmp)  {
			if (this->key != tmp.key||this->value != tmp.value) {
				return true;
			}
			else {
				return false;
			}

		}
	};

struct compare_key_value
	{
		__host__ __device__
			bool operator()(int lhs, int rhs) {
			return lhs < rhs;
		}
	};

void minmax_element_test() {
	const int N = 6;
	int data[N] = { 1, 0, 2, 2, 1, 3 };

//CHECK:  if (dpct::is_device_ptr(data)) {
//CHECK-NEXT:     oneapi::dpl::minmax_element(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(data), dpct::device_pointer<int>(data + N));
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::minmax_element(oneapi::dpl::execution::seq, data, data + N);
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(data)) {
//CHECK-NEXT:    oneapi::dpl::minmax_element(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(data), dpct::device_pointer<int>(data + N));
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::minmax_element(oneapi::dpl::execution::seq, data, data + N);
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(data)) {
//CHECK-NEXT:    oneapi::dpl::minmax_element(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(data), dpct::device_pointer<int>(data + N), compare_key_value());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::minmax_element(oneapi::dpl::execution::seq, data, data + N, compare_key_value());
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(data)) {
//CHECK-NEXT:    oneapi::dpl::minmax_element(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(data), dpct::device_pointer<int>(data + N), compare_key_value());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::minmax_element(oneapi::dpl::execution::seq, data, data + N, compare_key_value());
//CHECK-NEXT:  };
  thrust::minmax_element(thrust::host, data, data+N);
  thrust::minmax_element(data, data+N);
  thrust::minmax_element(thrust::host, data, data+N, compare_key_value());
  thrust::minmax_element(data, data+N, compare_key_value());
}


void is_sorted_test() {
    const int N=6;
    int datas[N]={1,4,2,8,5,7};

//CHECK:    std::vector<int> h_v(datas, datas + N);
//CHECK-NEXT:    dpct::device_vector<int> d_v(datas, datas + N);
//CHECK-NEXT:    std::greater<int> comp;
    thrust::host_vector<int> h_v(datas,datas+N);
    thrust::device_vector<int> d_v(datas,datas+N);
    thrust::greater<int> comp;

//CHECK:    if (dpct::is_device_ptr(datas)) {
//CHECK-NEXT:        oneapi::dpl::is_sorted(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(datas), dpct::device_pointer<int>(datas + N));
//CHECK-NEXT:    } else {
//CHECK-NEXT:        oneapi::dpl::is_sorted(oneapi::dpl::execution::seq, datas, datas + N);
//CHECK-NEXT:    };
//CHECK-NEXT:    if (dpct::is_device_ptr(datas)) {
//CHECK-NEXT:        oneapi::dpl::is_sorted(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(datas), dpct::device_pointer<int>(datas + N));
//CHECK-NEXT:    } else {
//CHECK-NEXT:        oneapi::dpl::is_sorted(oneapi::dpl::execution::seq, datas, datas + N);
//CHECK-NEXT:    };
//CHECK-NEXT:    if (dpct::is_device_ptr(datas)) {
//CHECK-NEXT:        oneapi::dpl::is_sorted(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(datas), dpct::device_pointer<int>(datas + N), comp);
//CHECK-NEXT:    } else {
//CHECK-NEXT:        oneapi::dpl::is_sorted(oneapi::dpl::execution::seq, datas, datas + N, comp);
//CHECK-NEXT:    };
//CHECK-NEXT:    if (dpct::is_device_ptr(datas)) {
//CHECK-NEXT:        oneapi::dpl::is_sorted(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(datas), dpct::device_pointer<int>(datas + N), comp);
//CHECK-NEXT:    } else {
//CHECK-NEXT:        oneapi::dpl::is_sorted(oneapi::dpl::execution::seq, datas, datas + N, comp);
//CHECK-NEXT:    };
    thrust::is_sorted(thrust::host, datas, datas+N);
    thrust::is_sorted( datas, datas+N);
    thrust::is_sorted(thrust::host,datas, datas+N,comp);
    thrust::is_sorted(datas, datas+N,comp);
}

struct is_even
{
  __host__ __device__
  bool operator()(const int &x)
  {
    return (x % 2) == 0;
  }
};


void is_partition_test() {
  int datas[]={1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int ans[]={2, 4, 6, 8, 10, 1, 3, 5, 7, 9};
  const int N=sizeof(datas)/sizeof(int);
  int stencil[N]={1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  thrust::host_vector<int> h_vdata(datas,datas+N);
  thrust::host_vector<int> h_vstencil(stencil,stencil+N);
  thrust::device_vector<int> d_v(datas,datas+N);
  thrust::host_vector<int> h_v(datas,datas+N);
  thrust::device_vector<int> d_vdata(datas,datas+N);
  thrust::device_vector<int> d_vstencil(stencil,stencil+N);

//CHECK:  if (dpct::is_device_ptr(datas)) {
//CHECK-NEXT:    oneapi::dpl::partition(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(datas), dpct::device_pointer<int>(datas + N), is_even());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::partition(oneapi::dpl::execution::seq, datas, datas + N, is_even());
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(datas)) {
//CHECK-NEXT:    oneapi::dpl::partition(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(datas), dpct::device_pointer<int>(datas + N), is_even());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::partition(oneapi::dpl::execution::seq, datas, datas + N, is_even());
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(datas)) {
//CHECK-NEXT:    dpct::partition(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(datas), dpct::device_pointer<int>(datas + N), dpct::device_pointer<int>(stencil), is_even());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    dpct::partition(oneapi::dpl::execution::seq, datas, datas + N, stencil, is_even());
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(datas)) {
//CHECK-NEXT:    dpct::partition(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(datas), dpct::device_pointer<int>(datas + N), dpct::device_pointer<int>(stencil), is_even());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    dpct::partition(oneapi::dpl::execution::seq, datas, datas + N, stencil, is_even());
//CHECK-NEXT:  };
  thrust::partition(thrust::host, datas, datas+N,is_even());
  thrust::partition( datas, datas+N, is_even());
  thrust::partition(thrust::host,  datas, datas+N, stencil,is_even());
  thrust::partition( datas, datas+N, stencil,is_even());
}

void unique_copy_test() {
  const int N=7;
  int A[N]={1, 3, 3, 3, 2, 2, 1};
  int B[N];
  const int M=N-3;
  int ans[M]={1, 3, 2, 1};

//CHECK:  if (dpct::is_device_ptr(A)) {
//CHECK-NEXT:    oneapi::dpl::unique_copy(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(A), dpct::device_pointer<int>(A + N), dpct::device_pointer<int>(B));
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::unique_copy(oneapi::dpl::execution::seq, A, A + N, B);
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(A)) {
//CHECK-NEXT:    oneapi::dpl::unique_copy(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(A), dpct::device_pointer<int>(A + N), dpct::device_pointer<int>(B));
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::unique_copy(oneapi::dpl::execution::seq, A, A + N, B);
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(A)) {
//CHECK-NEXT:    oneapi::dpl::unique_copy(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(A), dpct::device_pointer<int>(A + N), dpct::device_pointer<int>(B), oneapi::dpl::equal_to<int>());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::unique_copy(oneapi::dpl::execution::seq, A, A + N, B, oneapi::dpl::equal_to<int>());
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(A)) {
//CHECK-NEXT:    oneapi::dpl::unique_copy(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(A), dpct::device_pointer<int>(A + N), dpct::device_pointer<int>(B), oneapi::dpl::equal_to<int>());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::unique_copy(oneapi::dpl::execution::seq, A, A + N, B, oneapi::dpl::equal_to<int>());
//CHECK-NEXT:  };
  thrust::unique_copy(thrust::host,A, A + N, B);
  thrust::unique_copy(A, A + N, B);
  thrust::unique_copy(thrust::host,A, A + N, B, thrust::equal_to<int>());
  thrust::unique_copy(A, A + N, B, thrust::equal_to<int>());
}


void stable_sort_test() {
  const int N=6;
  int datas[N]={1, 4, 2, 8, 5, 7};
  int ans[N]={1, 2, 4, 5, 7, 8};

//CHECK:  if (dpct::is_device_ptr(datas)) {
//CHECK-NEXT:    oneapi::dpl::stable_sort(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(datas), dpct::device_pointer<int>(datas + N));
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::stable_sort(oneapi::dpl::execution::seq, datas, datas + N);
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(datas)) {
//CHECK-NEXT:    oneapi::dpl::stable_sort(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(datas), dpct::device_pointer<int>(datas + N));
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::stable_sort(oneapi::dpl::execution::seq, datas, datas + N);
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(datas)) {
//CHECK-NEXT:    oneapi::dpl::stable_sort(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(datas), dpct::device_pointer<int>(datas + N), std::greater<int>());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::stable_sort(oneapi::dpl::execution::seq, datas, datas + N, std::greater<int>());
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(datas)) {
//CHECK-NEXT:    oneapi::dpl::stable_sort(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(datas), dpct::device_pointer<int>(datas + N), std::greater<int>());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::stable_sort(oneapi::dpl::execution::seq, datas, datas + N, std::greater<int>());
//CHECK-NEXT:  };
  thrust::stable_sort(thrust::host, datas,datas+N);
  thrust::stable_sort(datas,datas+N);
  thrust::stable_sort(thrust::host, datas,datas+N, thrust::greater<int>());
  thrust::stable_sort(datas,datas+N, thrust::greater<int>());
}

void set_difference_by_key_test() {
  const int N=7,M=5,P=3;
  int Akey[N]={0, 1, 3, 4, 5, 6, 9};
  int Avalue[N]={0, 0, 0, 0, 0, 0, 0};
  int Bkey[M]={1, 3, 5, 7, 9};
  int Bvalue[N]={1, 1, 1, 1, 1 };

  int Ckey[P];
  int Cvalue[P];
  int anskey[P]={0,4,6};
  int ansvalue[P]={0,0,0};

//CHECK:  if (dpct::is_device_ptr(Akey)) {
//CHECK-NEXT:    dpct::set_difference(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(Akey), dpct::device_pointer<int>(Akey + N), dpct::device_pointer<int>(Bkey), dpct::device_pointer<int>(Bkey + M), dpct::device_pointer<int>(Avalue), dpct::device_pointer<int>(Bvalue), dpct::device_pointer<int>(Ckey), dpct::device_pointer<int>(Cvalue));
//CHECK-NEXT:  } else {
//CHECK-NEXT:    dpct::set_difference(oneapi::dpl::execution::seq, Akey, Akey + N, Bkey, Bkey + M, Avalue, Bvalue, Ckey, Cvalue);
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(Akey)) {
//CHECK-NEXT:    dpct::set_difference(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(Akey), dpct::device_pointer<int>(Akey + N), dpct::device_pointer<int>(Bkey), dpct::device_pointer<int>(Bkey + M), dpct::device_pointer<int>(Avalue), dpct::device_pointer<int>(Bvalue), dpct::device_pointer<int>(Ckey), dpct::device_pointer<int>(Cvalue));
//CHECK-NEXT:  } else {
//CHECK-NEXT:    dpct::set_difference(oneapi::dpl::execution::seq, Akey, Akey + N, Bkey, Bkey + M, Avalue, Bvalue, Ckey, Cvalue);
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(Akey)) {
//CHECK-NEXT:    dpct::set_difference(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(Akey), dpct::device_pointer<int>(Akey + N), dpct::device_pointer<int>(Bkey), dpct::device_pointer<int>(Bkey + M), dpct::device_pointer<int>(Avalue), dpct::device_pointer<int>(Bvalue), dpct::device_pointer<int>(Ckey), dpct::device_pointer<int>(Cvalue), std::greater<int>());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    dpct::set_difference(oneapi::dpl::execution::seq, Akey, Akey + N, Bkey, Bkey + M, Avalue, Bvalue, Ckey, Cvalue, std::greater<int>());
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(Akey)) {
//CHECK-NEXT:    dpct::set_difference(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(Akey), dpct::device_pointer<int>(Akey + N), dpct::device_pointer<int>(Bkey), dpct::device_pointer<int>(Bkey + M), dpct::device_pointer<int>(Avalue), dpct::device_pointer<int>(Bvalue), dpct::device_pointer<int>(Ckey), dpct::device_pointer<int>(Cvalue), std::greater<int>());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    dpct::set_difference(oneapi::dpl::execution::seq, Akey, Akey + N, Bkey, Bkey + M, Avalue, Bvalue, Ckey, Cvalue, std::greater<int>());
//CHECK-NEXT:  };
  thrust::set_difference_by_key(thrust::host,Akey,Akey+N,Bkey,Bkey+M,Avalue,Bvalue,Ckey,Cvalue);
  thrust::set_difference_by_key(Akey,Akey+N,Bkey,Bkey+M,Avalue,Bvalue,Ckey,Cvalue);
  thrust::set_difference_by_key(thrust::host,Akey,Akey+N,Bkey,Bkey+M,Avalue,Bvalue,Ckey,Cvalue, thrust::greater<int>());
  thrust::set_difference_by_key(Akey,Akey+N,Bkey,Bkey+M,Avalue,Bvalue,Ckey,Cvalue, thrust::greater<int>());
}


void set_difference_test() {
  const int N=7,M=5,P=3;
  int A[N]={0, 1, 3, 4, 5, 6, 9};
  int B[M]={1, 3, 5, 7, 9};
  int C[P];
  int ans[P]={0,4,6};

//CHECK:  if (dpct::is_device_ptr(A)) {
//CHECK-NEXT:    oneapi::dpl::set_difference(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(A), dpct::device_pointer<int>(A + N), dpct::device_pointer<int>(B), dpct::device_pointer<int>(B + M), dpct::device_pointer<int>(C));
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::set_difference(oneapi::dpl::execution::seq, A, A + N, B, B + M, C);
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(A)) {
//CHECK-NEXT:    oneapi::dpl::set_difference(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(A), dpct::device_pointer<int>(A + N), dpct::device_pointer<int>(B), dpct::device_pointer<int>(B + M), dpct::device_pointer<int>(C));
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::set_difference(oneapi::dpl::execution::seq, A, A + N, B, B + M, C);
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(A)) {
//CHECK-NEXT:    oneapi::dpl::set_difference(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(A), dpct::device_pointer<int>(A + N), dpct::device_pointer<int>(B), dpct::device_pointer<int>(B + M), dpct::device_pointer<int>(C), std::greater<int>());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::set_difference(oneapi::dpl::execution::seq, A, A + N, B, B + M, C, std::greater<int>());
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(A)) {
//CHECK-NEXT:    oneapi::dpl::set_difference(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(A), dpct::device_pointer<int>(A + N), dpct::device_pointer<int>(B), dpct::device_pointer<int>(B + M), dpct::device_pointer<int>(C), std::greater<int>());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::set_difference(oneapi::dpl::execution::seq, A, A + N, B, B + M, C, std::greater<int>());
//CHECK-NEXT:  };
  thrust::set_difference(thrust::host, A,A+N,B,B+M,C);
  thrust::set_difference( A,A+N,B,B+M,C);
  thrust::set_difference(thrust::host, A,A+N,B,B+M,C, thrust::greater<int>());
  thrust::set_difference( A,A+N,B,B+M,C, thrust::greater<int>());
}

void tabulate_test() {
  const int N=10;
  int A[N];
  int ans[N]={0, -1, -2, -3, -4, -5, -6, -7, -8, -9};
  thrust::host_vector<int> h_V(A,A+N);
  thrust::device_vector<int> d_V(A,A+N);

//CHECK:  if (dpct::is_device_ptr(A)) {
//CHECK-NEXT:    dpct::for_each_index(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(A), dpct::device_pointer<int>(A + N), std::negate<int>());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    dpct::for_each_index(oneapi::dpl::execution::seq, A, A + N, std::negate<int>());
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(A)) {
//CHECK-NEXT:    dpct::for_each_index(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(A), dpct::device_pointer<int>(A + N), std::negate<int>());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    dpct::for_each_index(oneapi::dpl::execution::seq, A, A + N, std::negate<int>());
//CHECK-NEXT:  };
  thrust::tabulate(thrust::host, A,A+N, thrust::negate<int>());
  thrust::tabulate(A,A+N, thrust::negate<int>());
}

struct add_functor
{
  void operator()(int & x)
  {
    x++;
  }
};
void for_each_n_test() {
  const int N=3;
  int A[N]={0,1,2};
  int ans[N]={1,2,3};
  thrust::host_vector<int> h_V(A,A+N);
  thrust::device_vector<int> d_V(A,A+N);

//CHECK:  if (dpct::is_device_ptr(A)) {
//CHECK-NEXT:    oneapi::dpl::for_each_n(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(A), N, add_functor());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::for_each_n(oneapi::dpl::execution::seq, A, N, add_functor());
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(N)) {
//CHECK-NEXT:    oneapi::dpl::for_each_n(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(A), N, add_functor());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::for_each_n(oneapi::dpl::execution::seq, A, N, add_functor());
//CHECK-NEXT:  };
  thrust::for_each_n(thrust::host, A, N, add_functor());
  thrust::for_each_n(A, N, add_functor());
}



void remove_copy_test() {
  const int N = 6;
  int V[N] = {-2, 0, -1, 0, 1, 2};
  int result[N - 2];

//CHECK:  if (dpct::is_device_ptr(V)) {
//CHECK-NEXT:    oneapi::dpl::remove_copy(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(V), dpct::device_pointer<int>(V + N), dpct::device_pointer<int>(result), 0);
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::remove_copy(oneapi::dpl::execution::seq, V, V + N, result, 0);
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(V)) {
//CHECK-NEXT:    oneapi::dpl::remove_copy(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(V), dpct::device_pointer<int>(V + N), dpct::device_pointer<int>(result), 0);
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::remove_copy(oneapi::dpl::execution::seq, V, V + N, result, 0);
//CHECK-NEXT:  };
  thrust::remove_copy(thrust::host, V, V + N, result, 0);
  thrust::remove_copy(V, V + N, result, 0);
}

void transform_exclusive_scan_test() {
  const int N=6;
  int A[N]={1, 0, 2, 2, 1, 3};
  int ans[N]={4, 3, 3, 1, -1, -2};
  thrust::negate<int> unary_op;
  thrust::plus<int> binary_op;

//CHECK:  if (dpct::is_device_ptr(A)) {
//CHECK-NEXT:    oneapi::dpl::transform_exclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(A), dpct::device_pointer<int>(A + N), dpct::device_pointer<int>(A), 4, binary_op, unary_op);
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::transform_exclusive_scan(oneapi::dpl::execution::seq, A, A + N, A, 4, binary_op, unary_op);
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(A)) {
//CHECK-NEXT:    oneapi::dpl::transform_exclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(A), dpct::device_pointer<int>(A + N), dpct::device_pointer<int>(A), 4, binary_op, unary_op);
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::transform_exclusive_scan(oneapi::dpl::execution::seq, A, A + N, A, 4, binary_op, unary_op);
//CHECK-NEXT:  };
  thrust::transform_exclusive_scan(thrust::host, A, A+N, A, unary_op, 4, binary_op);
  thrust::transform_exclusive_scan(A, A+N, A, unary_op, 4, binary_op);
}

void set_intersection_by_key_test() {
  const int N = 6, M = 7, P = 3;
  int Akey[N] = {1, 3, 5, 7, 9, 11};
  int Avalue[N] = {0, 0, 0, 0, 0, 0};
  int Bkey[M] = {1, 1, 2, 3, 5, 8, 13};

  int Ckey[P];
  int Cvalue[P];
  int anskey[P] = {1, 3, 5};
  int ansvalue[P] = {0, 0, 0};

  thrust::host_vector<int> h_VAkey(Akey, Akey + N);
  thrust::host_vector<int> h_VAvalue(Avalue, Avalue + N);

  thrust::host_vector<int> h_VBkey(Bkey, Bkey + M);

  thrust::host_vector<int> h_VCkey(Ckey, Ckey + P);
  thrust::host_vector<int> h_VCvalue(Cvalue, Cvalue + P);
  typedef thrust::pair<thrust::host_vector<int>::iterator,
                       thrust::host_vector<int>::iterator> iter_pair;
  thrust::device_vector<int> d_VAkey(Akey, Akey + N);
  thrust::device_vector<int> d_VAvalue(Avalue, Avalue + N);
  thrust::device_vector<int> d_VBkey(Bkey, Bkey + M);
  thrust::device_vector<int> d_VCkey(Ckey, Ckey + P);
  thrust::device_vector<int> d_VCvalue(Cvalue, Cvalue + P);

//CHECK:  if (dpct::is_device_ptr(Akey)) {
//CHECK-NEXT:    dpct::set_intersection(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(Akey), dpct::device_pointer<int>(Akey + N), dpct::device_pointer<int>(Bkey), dpct::device_pointer<int>(Bkey + M), dpct::device_pointer<int>(Avalue), dpct::device_pointer<int>(Ckey), dpct::device_pointer<int>(Cvalue));
//CHECK-NEXT:  } else {
//CHECK-NEXT:    dpct::set_intersection(oneapi::dpl::execution::seq, Akey, Akey + N, Bkey, Bkey + M, Avalue, Ckey, Cvalue);
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(Akey)) {
//CHECK-NEXT:    dpct::set_intersection(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(Akey), dpct::device_pointer<int>(Akey + N), dpct::device_pointer<int>(Bkey), dpct::device_pointer<int>(Bkey + M), dpct::device_pointer<int>(Avalue), dpct::device_pointer<int>(Ckey), dpct::device_pointer<int>(Cvalue));
//CHECK-NEXT:  } else {
//CHECK-NEXT:    dpct::set_intersection(oneapi::dpl::execution::seq, Akey, Akey + N, Bkey, Bkey + M, Avalue, Ckey, Cvalue);
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(Akey)) {
//CHECK-NEXT:    dpct::set_intersection(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(Akey), dpct::device_pointer<int>(Akey + N), dpct::device_pointer<int>(Bkey), dpct::device_pointer<int>(Bkey + M), dpct::device_pointer<int>(Avalue), dpct::device_pointer<int>(Ckey), dpct::device_pointer<int>(Cvalue), std::greater<int>());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    dpct::set_intersection(oneapi::dpl::execution::seq, Akey, Akey + N, Bkey, Bkey + M, Avalue, Ckey, Cvalue, std::greater<int>());
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(Akey)) {
//CHECK-NEXT:    dpct::set_intersection(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(Akey), dpct::device_pointer<int>(Akey + N), dpct::device_pointer<int>(Bkey), dpct::device_pointer<int>(Bkey + M), dpct::device_pointer<int>(Avalue), dpct::device_pointer<int>(Ckey), dpct::device_pointer<int>(Cvalue), std::greater<int>());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    dpct::set_intersection(oneapi::dpl::execution::seq, Akey, Akey + N, Bkey, Bkey + M, Avalue, Ckey, Cvalue, std::greater<int>());
//CHECK-NEXT:  };
  thrust::set_intersection_by_key(thrust::host, Akey, Akey + N, Bkey, Bkey + M, Avalue, Ckey, Cvalue);
  thrust::set_intersection_by_key(Akey, Akey + N, Bkey, Bkey + M, Avalue, Ckey, Cvalue);
  thrust::set_intersection_by_key(thrust::host, Akey, Akey + N, Bkey, Bkey + M, Avalue, Ckey, Cvalue, thrust::greater<int>());
  thrust::set_intersection_by_key(Akey, Akey + N, Bkey, Bkey + M, Avalue, Ckey, Cvalue, thrust::greater<int>());
}

void partition_copy_test() {
  int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int S[] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  int result[10];
  const int N = sizeof(data) / sizeof(int);
  int *evens = result;
  int *odds = result + N / 2;

//CHECK:  if (dpct::is_device_ptr(data)) {
//CHECK-NEXT:    oneapi::dpl::partition_copy(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(data), dpct::device_pointer<int>(data + N), dpct::device_pointer<int>(evens), dpct::device_pointer<int>(odds), is_even());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::partition_copy(oneapi::dpl::execution::seq, data, data + N, evens, odds, is_even());
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(data)) {
//CHECK-NEXT:    oneapi::dpl::partition_copy(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(data), dpct::device_pointer<int>(data + N), dpct::device_pointer<int>(evens), dpct::device_pointer<int>(odds), is_even());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::partition_copy(oneapi::dpl::execution::seq, data, data + N, evens, odds, is_even());
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(data)) {
//CHECK-NEXT:    dpct::partition_copy(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(data), dpct::device_pointer<int>(data + N), dpct::device_pointer<int>(S), dpct::device_pointer<int>(evens), dpct::device_pointer<int>(odds), is_even());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    dpct::partition_copy(oneapi::dpl::execution::seq, data, data + N, S, evens, odds, is_even());
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(data)) {
//CHECK-NEXT:    dpct::partition_copy(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(data), dpct::device_pointer<int>(data + N), dpct::device_pointer<int>(S), dpct::device_pointer<int>(evens), dpct::device_pointer<int>(odds), is_even());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    dpct::partition_copy(oneapi::dpl::execution::seq, data, data + N, S, evens, odds, is_even());
//CHECK-NEXT:  };
  thrust::partition_copy(thrust::host, data, data + N, evens, odds, is_even());
  thrust::partition_copy(data, data + N, evens, odds, is_even());
  thrust::partition_copy(thrust::host, data, data + N, S, evens, odds, is_even());
  thrust::partition_copy(data, data + N, S, evens, odds, is_even());
}

void stable_partition_copy_test() {
  int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int S[] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  int result[10];
  const int N = sizeof(data) / sizeof(int);
  int *evens = result;
  int *odds = result + N / 2;

//CHECK:  if (dpct::is_device_ptr(data)) {
//CHECK-NEXT:    dpct::stable_partition_copy(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(data), dpct::device_pointer<int>(data + N), dpct::device_pointer<int>(evens), dpct::device_pointer<int>(odds), is_even());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    dpct::stable_partition_copy(oneapi::dpl::execution::seq, data, data + N, evens, odds, is_even());
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(data)) {
//CHECK-NEXT:    dpct::stable_partition_copy(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(data), dpct::device_pointer<int>(data + N), dpct::device_pointer<int>(evens), dpct::device_pointer<int>(odds), is_even());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    dpct::stable_partition_copy(oneapi::dpl::execution::seq, data, data + N, evens, odds, is_even());
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(data)) {
//CHECK-NEXT:    dpct::stable_partition_copy(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(data), dpct::device_pointer<int>(data + N), dpct::device_pointer<int>(S), dpct::device_pointer<int>(evens), dpct::device_pointer<int>(odds), is_even());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    dpct::stable_partition_copy(oneapi::dpl::execution::seq, data, data + N, S, evens, odds, is_even());
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(data)) {
//CHECK-NEXT:    dpct::stable_partition_copy(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(data), dpct::device_pointer<int>(data + N), dpct::device_pointer<int>(S), dpct::device_pointer<int>(evens), dpct::device_pointer<int>(odds), is_even());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    dpct::stable_partition_copy(oneapi::dpl::execution::seq, data, data + N, S, evens, odds, is_even());
//CHECK-NEXT:  };
  thrust::stable_partition_copy(thrust::host, data, data + N, evens, odds, is_even());
  thrust::stable_partition_copy(data, data + N, evens, odds, is_even());
  thrust::stable_partition_copy(thrust::host, data, data + N, S, evens, odds, is_even());
  thrust::stable_partition_copy(data, data + N, S, evens, odds, is_even());
}

void stable_partition_test() {

  int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int S[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const int N = sizeof(data) / sizeof(int);


//CHECK:  if (dpct::is_device_ptr(data)) {
//CHECK-NEXT:    oneapi::dpl::stable_partition(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(data), dpct::device_pointer<int>(data + N), is_even());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::stable_partition(oneapi::dpl::execution::seq, data, data + N, is_even());
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(data)) {
//CHECK-NEXT:    oneapi::dpl::stable_partition(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(data), dpct::device_pointer<int>(data + N), is_even());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::stable_partition(oneapi::dpl::execution::seq, data, data + N, is_even());
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(data)) {
//CHECK-NEXT:    dpct::stable_partition(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(data), dpct::device_pointer<int>(data + N), dpct::device_pointer<int>(S), is_even());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    dpct::stable_partition(oneapi::dpl::execution::seq, data, data + N, S, is_even());
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(data)) {
//CHECK-NEXT:    dpct::stable_partition(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(data), dpct::device_pointer<int>(data + N), dpct::device_pointer<int>(S), is_even());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    dpct::stable_partition(oneapi::dpl::execution::seq, data, data + N, S, is_even());
//CHECK-NEXT:  };
  thrust::stable_partition(thrust::host, data, data + N, is_even());
  thrust::stable_partition(data, data + N, is_even());
  thrust::stable_partition(thrust::host, data, data + N, S, is_even());
  thrust::stable_partition(data, data + N, S, is_even());
}



void remvoe_test() {
  const int N = 6;
  int data[N] = {3, 1, 4, 1, 5, 9};

//CHECK:  if (dpct::is_device_ptr(data)) {
//CHECK-NEXT:    oneapi::dpl::remove(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(data), dpct::device_pointer<int>(data + N), 1);
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::remove(oneapi::dpl::execution::seq, data, data + N, 1);
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(data)) {
//CHECK-NEXT:    oneapi::dpl::remove(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(data), dpct::device_pointer<int>(data + N), 1);
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::remove(oneapi::dpl::execution::seq, data, data + N, 1);
//CHECK-NEXT:  };
  thrust::remove(thrust::host, data, data + N, 1);
  thrust::remove(data, data + N, 1);
}

struct greater_than_four {
  __host__ __device__ bool operator()(int x) const { return x > 4; }
};

void find_if_test() {
  const int N = 4;
  int data[4] = {0,5, 3, 7};

//CHECK:  if (dpct::is_device_ptr(data)) {
//CHECK-NEXT:    oneapi::dpl::find_if(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(data), dpct::device_pointer<int>(data + 3), greater_than_four());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::find_if(oneapi::dpl::execution::seq, data, data + 3, greater_than_four());
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(data)) {
//CHECK-NEXT:    oneapi::dpl::find_if(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(data), dpct::device_pointer<int>(data + 3), greater_than_four());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::find_if(oneapi::dpl::execution::seq, data, data + 3, greater_than_four());
//CHECK-NEXT:  };
  thrust::find_if(data, data+3, greater_than_four());
  thrust::find_if(thrust::host, data, data+3, greater_than_four());
}

void find_if_not_test() {
  const int N = 4;
  int data[4] = {0,5, 3, 7};

//CHECK:  if (dpct::is_device_ptr(data)) {
//CHECK-NEXT:    oneapi::dpl::find_if_not(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(data), dpct::device_pointer<int>(data + 3), greater_than_four());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::find_if_not(oneapi::dpl::execution::seq, data, data + 3, greater_than_four());
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(data)) {
//CHECK-NEXT:    oneapi::dpl::find_if_not(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(data), dpct::device_pointer<int>(data + 3), greater_than_four());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::find_if_not(oneapi::dpl::execution::seq, data, data + 3, greater_than_four());
//CHECK-NEXT:  };
  thrust::find_if_not(data, data+3, greater_than_four());
  thrust::find_if_not(thrust::host, data, data+3, greater_than_four());
}

void mismatch_test() {
  const int N = 4;
  int A[N] = {0, 5, 3, 7};
  int B[N] = {0, 5, 8, 7};

//CHECK:  if (dpct::is_device_ptr(A)) {
//CHECK-NEXT:    oneapi::dpl::mismatch(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(A), dpct::device_pointer<int>(A + N), dpct::device_pointer<int>(B));
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::mismatch(oneapi::dpl::execution::seq, A, A + N, B);
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(A)) {
//CHECK-NEXT:    oneapi::dpl::mismatch(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(A), dpct::device_pointer<int>(A + N), dpct::device_pointer<int>(B));
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::mismatch(oneapi::dpl::execution::seq, A, A + N, B);
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(A)) {
//CHECK-NEXT:    oneapi::dpl::mismatch(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(A), dpct::device_pointer<int>(A + N), dpct::device_pointer<int>(B), oneapi::dpl::equal_to<int>());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::mismatch(oneapi::dpl::execution::seq, A, A + N, B, oneapi::dpl::equal_to<int>());
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(A)) {
//CHECK-NEXT:    oneapi::dpl::mismatch(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(A), dpct::device_pointer<int>(A + N), dpct::device_pointer<int>(B), oneapi::dpl::equal_to<int>());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::mismatch(oneapi::dpl::execution::seq, A, A + N, B, oneapi::dpl::equal_to<int>());
//CHECK-NEXT:  };
  thrust::mismatch(thrust::host, A, A+N, B);
  thrust::mismatch( A, A+N, B);
  thrust::mismatch(thrust::host, A, A+N, B, thrust::equal_to<int>());
  thrust::mismatch( A, A+N, B, thrust::equal_to<int>());
}

void replace_copy_test() {
  const int N = 4;
  int data[] = {1, 2, 3, 1};
  int result[N];

//CHECK:  if (dpct::is_device_ptr(data)) {
//CHECK-NEXT:    oneapi::dpl::replace_copy(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(data), dpct::device_pointer<int>(data + N), dpct::device_pointer<int>(result), 1, 99);
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::replace_copy(oneapi::dpl::execution::seq, data, data + N, result, 1, 99);
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(data)) {
//CHECK-NEXT:    oneapi::dpl::replace_copy(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(data), dpct::device_pointer<int>(data + N), dpct::device_pointer<int>(result), 1, 99);
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::replace_copy(oneapi::dpl::execution::seq, data, data + N, result, 1, 99);
//CHECK-NEXT:  };
  thrust::replace_copy(thrust::host, data, data + N, result, 1, 99);
  thrust::replace_copy(data, data + N, result, 1, 99);
}

void reverse() {
  const int N = 6;
  int data[N] = {0, 1, 2, 3, 4, 5};

//CHECK:  if (dpct::is_device_ptr(data)) {
//CHECK-NEXT:    oneapi::dpl::reverse(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(data), dpct::device_pointer<int>(data + N));
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::reverse(oneapi::dpl::execution::seq, data, data + N);
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(data)) {
//CHECK-NEXT:    oneapi::dpl::reverse(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(data), dpct::device_pointer<int>(data + N));
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::reverse(oneapi::dpl::execution::seq, data, data + N);
//CHECK-NEXT:  };
  thrust::reverse(thrust::host, data, data + N);
  thrust::reverse(data, data + N);
}

void equal_range() {
  int data[] = {0, 2, 5, 7, 8};
  const int N = 5;

//CHECK:  if (dpct::is_device_ptr(data)) {
//CHECK-NEXT:    dpct::equal_range(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(data), dpct::device_pointer<int>(data + N), 0);
//CHECK-NEXT:  } else {
//CHECK-NEXT:    dpct::equal_range(oneapi::dpl::execution::seq, data, data + N, 0);
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(data)) {
//CHECK-NEXT:    dpct::equal_range(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(data), dpct::device_pointer<int>(data + N), 0);
//CHECK-NEXT:  } else {
//CHECK-NEXT:    dpct::equal_range(oneapi::dpl::execution::seq, data, data + N, 0);
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(data)) {
//CHECK-NEXT:    dpct::equal_range(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(data), dpct::device_pointer<int>(data + N), 0, oneapi::dpl::less<int>());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    dpct::equal_range(oneapi::dpl::execution::seq, data, data + N, 0, oneapi::dpl::less<int>());
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(data)) {
//CHECK-NEXT:    dpct::equal_range(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(data), dpct::device_pointer<int>(data + N), 0, oneapi::dpl::less<int>());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    dpct::equal_range(oneapi::dpl::execution::seq, data, data + N, 0, oneapi::dpl::less<int>());
//CHECK-NEXT:  };
  thrust::equal_range(thrust::host, data, data + N, 0); 
  thrust::equal_range(data, data + N, 0);
  thrust::equal_range(thrust::host, data, data + N, 0, thrust::less<int>()); 
  thrust::equal_range(data, data + N, 0, thrust::less<int>());
}

void transform_inclusive_scan() {
  const int N = 6;
  int data[6] = {1, 0, 2, 2, 1, 3};
  thrust::negate<int> unary_op;
  thrust::plus<int> binary_op;

//CHECK:  if (dpct::is_device_ptr(data)) {
//CHECK-NEXT:    oneapi::dpl::transform_inclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(data), dpct::device_pointer<int>(data + N), dpct::device_pointer<int>(data), binary_op, unary_op);
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::transform_inclusive_scan(oneapi::dpl::execution::seq, data, data + N, data, binary_op, unary_op);
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(data)) {
//CHECK-NEXT:    oneapi::dpl::transform_inclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(data), dpct::device_pointer<int>(data + N), dpct::device_pointer<int>(data), binary_op, unary_op);
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::transform_inclusive_scan(oneapi::dpl::execution::seq, data, data + N, data, binary_op, unary_op);
//CHECK-NEXT:  };
  thrust::transform_inclusive_scan(data, data + N, data, unary_op, binary_op);
  thrust::transform_inclusive_scan(thrust::host, data, data + N, data, unary_op, binary_op);
}
