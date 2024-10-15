// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasGetStatusString | FileCheck %s -check-prefix=cublasGetStatusString
// cublasGetStatusString: CUDA API:
// cublasGetStatusString-NEXT:   res /*const char **/ = cublasGetStatusString(status /*cublasStatus_t*/);
// cublasGetStatusString-NEXT: Is migrated to:
// cublasGetStatusString-NEXT:   res /*const char **/ = dpct::get_error_dummy(status);
