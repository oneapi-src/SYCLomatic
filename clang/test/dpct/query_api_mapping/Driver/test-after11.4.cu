// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3

/// Device Management

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuDeviceGetUuid_v2 | FileCheck %s -check-prefix=CUDEVICEGETUUID_V2
// CUDEVICEGETUUID_V2: CUDA API:
// CUDEVICEGETUUID_V2-NEXT:   cuDeviceGetUuid_v2(pu /*CUuuid **/, d /*CUdevice*/);
// CUDEVICEGETUUID_V2-NEXT: Is migrated to:
// CUDEVICEGETUUID_V2-NEXT:   *pu = dpct::get_device(d).get_device_info().get_uuid();
