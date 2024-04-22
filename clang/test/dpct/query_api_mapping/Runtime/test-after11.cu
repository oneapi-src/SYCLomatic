// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2

/// Stream Management

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaCtxResetPersistingL2Cache | FileCheck %s -check-prefix=CUDACTXRESETPERSISTINGL2CACHE
// CUDACTXRESETPERSISTINGL2CACHE: CUDA API:
// CUDACTXRESETPERSISTINGL2CACHE-NEXT:   cudaCtxResetPersistingL2Cache();
// CUDACTXRESETPERSISTINGL2CACHE-NEXT: The API is Removed.
// CUDACTXRESETPERSISTINGL2CACHE-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaStreamGetAttribute | FileCheck %s -check-prefix=CUDASTREAMGETATTRIBUTE
// CUDASTREAMGETATTRIBUTE: CUDA API:
// CUDASTREAMGETATTRIBUTE-NEXT:   cudaStreamGetAttribute(s /*cudaStream_t*/, a /*cudaStreamAttrID*/,
// CUDASTREAMGETATTRIBUTE-NEXT:                          pv /*cudaStreamAttrValue **/);
// CUDASTREAMGETATTRIBUTE-NEXT: The API is Removed.
// CUDASTREAMGETATTRIBUTE-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaStreamSetAttribute | FileCheck %s -check-prefix=CUDASTREAMSETATTRIBUTE
// CUDASTREAMSETATTRIBUTE: CUDA API:
// CUDASTREAMSETATTRIBUTE-NEXT:   cudaStreamSetAttribute(s /*cudaStream_t*/, a /*cudaStreamAttrID*/,
// CUDASTREAMSETATTRIBUTE-NEXT:                          pv /*cudaStreamAttrValue **/);
// CUDASTREAMSETATTRIBUTE-NEXT: The API is Removed.
// CUDASTREAMSETATTRIBUTE-EMPTY:
