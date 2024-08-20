// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2

/// Context Management

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuCtxResetPersistingL2Cache | FileCheck %s -check-prefix=CUCTXRESETPERSISTINGL2CACHE
// CUCTXRESETPERSISTINGL2CACHE: CUDA API:
// CUCTXRESETPERSISTINGL2CACHE-NEXT:   cuCtxResetPersistingL2Cache();
// CUCTXRESETPERSISTINGL2CACHE-NEXT: The API is Removed.
// CUCTXRESETPERSISTINGL2CACHE-EMPTY:
