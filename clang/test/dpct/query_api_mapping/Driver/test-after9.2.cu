// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1
// UNSUPPORTED: v8.0, v9.0, v9.1

/// Device Management

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuDeviceGetUuid | FileCheck %s -check-prefix=CUDEVICEGETUUID
// CUDEVICEGETUUID: CUDA API:
// CUDEVICEGETUUID-NEXT:   cuDeviceGetUuid(pu /*CUuuid **/, d /*CUdevice*/);
// CUDEVICEGETUUID-NEXT: Is migrated to:
// CUDEVICEGETUUID-NEXT:   *pu = dpct::get_device(d).get_device_info().get_uuid();
