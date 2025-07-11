// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8, cuda-12.0, cuda-12.1, cuda-12.2, cuda-12.3, cuda-12.4
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8, v12.0, v12.1, v12.2, v12.3, v12.4

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuCtxCreate_v3 | FileCheck %s -check-prefix=CUCTXCREATE_V3
// CUCTXCREATE_V3:  CUDA API:
// CUCTXCREATE_V3-NEXT:    cuCtxCreate_v3(ctx /*CUcontext **/, params_array /*CUexecAffinityParam **/, num /*int*/, flags /*unsigned int*/, device /*CUdevice*/);
// CUCTXCREATE_V3-NEXT:  Is migrated to:
// CUCTXCREATE_V3-NEXT:    *ctx = dpct::push_device_for_curr_thread(device);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuCtxCreate_v4 | FileCheck %s -check-prefix=CUCTXCREATE_V4
// CUCTXCREATE_V4:  CUDA API:
// CUCTXCREATE_V4-NEXT:    cuCtxCreate_v4(ctx /*CUcontext **/, params_array /*CUctxCreateParams **/, flags /*unsigned int*/, device /*CUdevice*/);
// CUCTXCREATE_V4-NEXT:  Is migrated to:
// CUCTXCREATE_V4-NEXT:    *ctx = dpct::push_device_for_curr_thread(device);
