// UNSUPPORTED: cuda-8.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8
// UNSUPPORTED: v8.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaGetDeviceProperties_v2 | FileCheck %s -check-prefix=cudaGetDeviceProperties_v2
// cudaGetDeviceProperties_v2: CUDA API:
// cudaGetDeviceProperties_v2-NEXT:   cudaDeviceProp *pd;
// cudaGetDeviceProperties_v2-NEXT:   cudaGetDeviceProperties_v2(pd /*cudaDeviceProp* */, i /*int*/);
// cudaGetDeviceProperties_v2-NEXT: Is migrated to:
// cudaGetDeviceProperties_v2-NEXT:   dpct::device_info *pd;
// cudaGetDeviceProperties_v2-NEXT:   dpct::get_device(i).get_device_info(*pd);
