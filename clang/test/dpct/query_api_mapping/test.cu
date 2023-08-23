// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=hfma | FileCheck %s

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=_hfma | FileCheck %s

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=HFMA | FileCheck %s

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__HfMa | FileCheck %s

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=" __HfMa " | FileCheck %s

// CHECK: CUDA API:
// CHECK-NEXT:   __hfma(h1 /*__half*/, h2 /*__half*/, h3 /*__half*/);
// CHECK-NEXT:   __hfma(b1 /*__nv_bfloat16*/, b2 /*__nv_bfloat16*/, b3 /*__nv_bfloat16*/);
// CHECK-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// CHECK-NEXT:   sycl::ext::intel::math::hfma(h1, h2);
// CHECK-NEXT:   sycl::ext::oneapi::experimental::fma(b1, b2, b3);
