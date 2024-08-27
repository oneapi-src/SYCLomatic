// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasRotEx | FileCheck %s -check-prefix=cublasRotEx
// cublasRotEx: CUDA API:
// cublasRotEx-NEXT:   cublasRotEx(handle /*cublasHandle_t*/, n /*int*/, x /*void **/,
// cublasRotEx-NEXT:               xtype /*cudaDataType*/, incx /*int*/, y /*void **/,
// cublasRotEx-NEXT:               ytype /*cudaDataType*/, incy /*int*/, c /*const void **/,
// cublasRotEx-NEXT:               s /*const void **/, cstype /*cudaDataType*/,
// cublasRotEx-NEXT:               computetype /*cudaDataType*/);
// cublasRotEx-NEXT: Is migrated to:
// cublasRotEx-NEXT:   dpct::blas::rot(handle, n, x, xtype, incx, y, ytype, incy, c, s, cstype);
