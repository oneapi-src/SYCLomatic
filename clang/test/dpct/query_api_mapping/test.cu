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

// RUN: not dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=aaa 2>&1 | FileCheck %s -check-prefix=NO_MAPPING
// NO_MAPPING: dpct exited with code: -43 (Error: The API mapping query for this API is not available yet. You may get the API mapping by migrating sample code from this CUDA API to the SYCL API with the tool.)

// CUDA 11 and after not have hdiv().
// RUN: not dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=hdiv 2>&1 | FileCheck %s -check-prefix=WRONG_CUDA_HEADER
// WRONG_CUDA_HEADER: dpct exited with code: -44 (Error: Can not find 'hdiv' in current CUDA header file: {{(.+)}}. Please check the API name or use a different CUDA header file with option "--cuda-include-path".)

// RUN: not dpct --cuda-include-path=%S --query-api-mapping=ncclBroadcast 2>&1 | FileCheck %s -check-prefix=NO_CUDA_HEADER
// NO_CUDA_HEADER: dpct exited with code: -45 (Error: Cannot find 'ncclBroadcast' in current CUDA header file: {{(.+)}}. Please specify the header file for 'ncclBroadcast' with option "--extra-arg".)
