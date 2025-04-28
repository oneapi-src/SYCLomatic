// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSetWorkspace | FileCheck %s -check-prefix=cublasSetWorkspace
// cublasSetWorkspace: CUDA API:
// cublasSetWorkspace-NEXT:   cublasSetWorkspace(handle /*cublasHandle_t*/, workspace /*void **/,
// cublasSetWorkspace-NEXT:                      size /*size_t*/);
// cublasSetWorkspace-NEXT: The API is Removed.
