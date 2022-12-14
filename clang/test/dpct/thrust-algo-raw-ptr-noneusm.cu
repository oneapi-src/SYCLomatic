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
			bool operator()(key_value lhs, key_value rhs) {
			return lhs.key < rhs.key;
		}
	};

void minmax_element_test() {
	const int N = 6;
	int data[N] = { 1, 0, 2, 2, 1, 3 };

//CHECK:  if (dpct::is_device_ptr(data)) {
//CHECK-NEXT:    oneapi::dpl::minmax_element(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(data + N));
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::minmax_element(oneapi::dpl::execution::seq, data, data + N);
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(data + N)) {
//CHECK-NEXT:    oneapi::dpl::minmax_element(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(data), dpct::device_pointer<int>(data + N));
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::minmax_element(oneapi::dpl::execution::seq, data, data + N);
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(data)) {
//CHECK-NEXT:    oneapi::dpl::minmax_element(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(data), dpct::device_pointer<int>(data + N), compare_key_value());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::minmax_element(oneapi::dpl::execution::seq, data, data + N, compare_key_value());
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(data + N)) {
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
//CHECK-NEXT:        oneapi::dpl::is_sorted(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(datas + N));
//CHECK-NEXT:    } else {
//CHECK-NEXT:        oneapi::dpl::is_sorted(oneapi::dpl::execution::seq, datas, datas + N);
//CHECK-NEXT:    };
//CHECK-NEXT:    if (dpct::is_device_ptr(datas + N)) {
//CHECK-NEXT:        oneapi::dpl::is_sorted(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(datas), dpct::device_pointer<int>(datas + N));
//CHECK-NEXT:    } else {
//CHECK-NEXT:        oneapi::dpl::is_sorted(oneapi::dpl::execution::seq, datas, datas + N);
//CHECK-NEXT:    };
//CHECK-NEXT:    if (dpct::is_device_ptr(datas)) {
//CHECK-NEXT:        oneapi::dpl::is_sorted(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(datas), dpct::device_pointer<int>(datas + N), comp);
//CHECK-NEXT:    } else {
//CHECK-NEXT:        oneapi::dpl::is_sorted(oneapi::dpl::execution::seq, datas, datas + N, comp);
//CHECK-NEXT:    };
//CHECK-NEXT:    if (dpct::is_device_ptr(datas + N)) {
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
//CHECK-NEXT:    oneapi::dpl::partition(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(datas + N, is_even()));
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::partition(oneapi::dpl::execution::seq, datas, datas + N, is_even());
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(datas + N)) {
//CHECK-NEXT:    oneapi::dpl::partition(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(datas), dpct::device_pointer<int>(datas + N), dpct::device_pointer<int>(is_even()));
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::partition(oneapi::dpl::execution::seq, datas, datas + N, is_even());
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(datas)) {
//CHECK-NEXT:    oneapi::dpl::partition(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(datas), dpct::device_pointer<int>(datas + N), dpct::device_pointer<>(h_stencil), is_even());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::partition(oneapi::dpl::execution::seq, datas, datas + N, h_stencil, is_even());
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(datas + N)) {
//CHECK-NEXT:    oneapi::dpl::partition(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(datas), dpct::device_pointer<int>(datas + N), dpct::device_pointer<>(h_stencil), is_even());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::partition(oneapi::dpl::execution::seq, datas, datas + N, h_stencil, is_even());
//CHECK-NEXT:  };
  thrust::partition(thrust::host, datas, datas+N,is_even());
  thrust::partition( datas, datas+N,is_even());
  thrust::partition(thrust::host,  datas, datas+N,h_stencil,is_even());
  thrust::partition( datas, datas+N,h_stencil,is_even());
}

void unique_copy_test() {
  const int N=7;
  int A[N]={1, 3, 3, 3, 2, 2, 1};
  int B[N];
  const int M=N-3;
  int ans[M]={1, 3, 2, 1};

//CHECK:  if (dpct::is_device_ptr(A)) {
//CHECK-NEXT:    oneapi::dpl::unique_copy(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(A + N, B));
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::unique_copy(oneapi::dpl::execution::seq, A, A + N, B);
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(A + N)) {
//CHECK-NEXT:    oneapi::dpl::unique_copy(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(A), dpct::device_pointer<int>(A + N), dpct::device_pointer<int>(B));
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::unique_copy(oneapi::dpl::execution::seq, A, A + N, B);
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(A)) {
//CHECK-NEXT:    oneapi::dpl::unique_copy(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(A), dpct::device_pointer<int>(A + N), dpct::device_pointer<int>(B), oneapi::dpl::equal_to<int>());
//CHECK-NEXT:  } else {
//CHECK-NEXT:    oneapi::dpl::unique_copy(oneapi::dpl::execution::seq, A, A + N, B, oneapi::dpl::equal_to<int>());
//CHECK-NEXT:  };
//CHECK-NEXT:  if (dpct::is_device_ptr(A + N)) {
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

  thrust::stable_sort(thrust::host, datas,datas+N);
  thrust::stable_sort(datas,datas+N);
  thrust::stable_sort(thrust::host, datas,datas+N, thrust::greater<int>());
  thrust::stable_sort(datas,datas+N, thrust::greater<int>());
}