// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0

/// Execution Control

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuFuncSetAttribute | FileCheck %s -check-prefix=CUFUNCSETATTRIBUTE
// CUFUNCSETATTRIBUTE: CUDA API:
// CUFUNCSETATTRIBUTE-NEXT:   cuFuncSetAttribute(f /*CUfunction*/, fa /*CUfunction_attribute*/, i /*int*/);
// CUFUNCSETATTRIBUTE-NEXT: The API is Removed.
// CUFUNCSETATTRIBUTE-EMPTY:
