// UNSUPPORTED: cuda-8.0, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, 
// UNSUPPORTED: v8.0, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1


// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::device_delete --extra-arg="-std=c++14"| FileCheck %s -check-prefix=device_delete
// device_delete: dpct::device_delete(d_array1, N);

