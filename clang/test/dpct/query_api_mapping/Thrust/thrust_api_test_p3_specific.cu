// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v12.9
// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=thrust::device_free --extra-arg="-std=c++14"| FileCheck %s -check-prefix=thrust_device_free
// thrust_device_free:CUDA API:
// thrust_device_free-NEXT:  thrust::device_ptr<thrust::complex<double>> d_ptr = thrust::device_malloc<thrust::complex<double>>(1);
// thrust_device_free-NEXT:  thrust::device_free(d_ptr);
// thrust_device_free-NEXT:Is migrated to:
// thrust_device_free-NEXT:  dpct::device_pointer<std::complex<double>> d_ptr = dpct::malloc_device<std::complex<double>>(1);
// thrust_device_free-NEXT:  dpct::free_device(d_ptr);