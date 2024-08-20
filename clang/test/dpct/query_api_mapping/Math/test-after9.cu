// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0

/// Half Precision Conversion and Data Movement

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_half2 | FileCheck %s -check-prefix=MAKE_HALF2
// MAKE_HALF2: CUDA API:
// MAKE_HALF2-NEXT:   make_half2(h1 /*__half*/, h2 /*__half*/);
// MAKE_HALF2-NEXT: Is migrated to:
// MAKE_HALF2-NEXT:   sycl::half2(h1, h2);
