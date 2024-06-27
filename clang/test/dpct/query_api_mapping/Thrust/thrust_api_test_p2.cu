// UNSUPPORTED: system-windows
// UNSUPPORTED: cuda-8.0, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, 

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::device_delete --extra-arg="-std=c++14"| FileCheck %s -check-prefix=device_delete
// device_delete: dpct::device_delete(d_array1, N);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::device_new --extra-arg="-std=c++14"| FileCheck %s -check-prefix=device_new
// device_new: /*1*/ dpct::device_pointer<int> d_array1 = dpct::device_new<int>(d_mem, N);
// device_new-NEXT:  /*2*/ dpct::device_pointer<int> d_array2 =
// device_new-NEXT:      dpct::device_new<int>(d_mem, val, N);
// device_new-NEXT:  /*3*/ dpct::device_pointer<int> d_array3 = dpct::device_new<int>(N);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::device_ptr --extra-arg="-std=c++14"| FileCheck %s -check-prefix=device_ptr
// device_ptr: dpct::device_pointer<int> d_mem = dpct::malloc_device<int>(N);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::malloc --extra-arg="-std=c++14"| FileCheck %s -check-prefix=malloc
// malloc:  /*1*/ dpct::tagged_pointer<int, dpct::device_sys_tag> ptr =
// malloc-NEXT:      dpct::malloc<int>(device_sys, N);
// malloc-NEXT:  /*2*/ dpct::tagged_pointer<void, dpct::device_sys_tag> void_ptr =
// malloc-NEXT:      dpct::malloc(device_sys, N);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::partition_point --extra-arg="-std=c++14"| FileCheck %s -check-prefix=partition_point
// partition_point:  /*1*/  dpct::partition_point(oneapi::dpl::execution::seq, h_v.begin(), h_v.end(), up);
// partition_point-NEXT:  /*2*/  dpct::partition_point(oneapi::dpl::execution::make_device_policy(q_ct1), d_v.begin(), d_v.end(), up);
// partition_point-NEXT:  /*3*/  dpct::partition_point(oneapi::dpl::execution::seq, data, data + 10, up);
// partition_point-NEXT:  /*4*/  dpct::partition_point(oneapi::dpl::execution::seq, h_v.begin(), h_v.end(), up);
// partition_point-NEXT:  /*5*/  dpct::partition_point(oneapi::dpl::execution::make_device_policy(q_ct1), d_v.begin(), d_v.end(), up);
// partition_point-NEXT:  /*6*/  dpct::partition_point(oneapi::dpl::execution::seq, data, data + 10, up);
