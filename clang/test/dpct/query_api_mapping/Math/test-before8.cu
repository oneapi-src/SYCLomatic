// REQUIRES: cuda-8.0
// REQUIRES: v8.0

/// Half Arithmetic Functions

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=hdiv | FileCheck %s -check-prefix=HDIV2
// HDIV2: CUDA API:
// HDIV2-NEXT:   hdiv(h1 /*__half*/, h2 /*__half*/);
// HDIV2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// HDIV2-NEXT:   sycl::ext::intel::math::hdiv(h1, h2);

/// Half2 Arithmetic Functions

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=h2div | FileCheck %s -check-prefix=H2DIV2
// H2DIV2: CUDA API:
// H2DIV2-NEXT:   h2div(h1 /*__half*/, h2 /*__half*/);
// H2DIV2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// H2DIV2-NEXT:   sycl::ext::intel::math::h2div(h1, h2);
