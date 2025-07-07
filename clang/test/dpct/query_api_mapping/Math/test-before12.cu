// UNSUPPORTED: cuda-12.0, cuda-12.1, cuda-12.2, cuda-12.3, cuda-12.4, cuda-12.5, cuda-12.6, cuda-12.7, cuda-12.8, cuda-12.9
// UNSUPPORTED: v12.0, v12.1, v12.2, v12.3, v12.4, v12.5, v12.6, v12.7, v12.8, v12.9

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=saturate | FileCheck %s -check-prefix=saturate
// saturate: CUDA API:
// saturate-NEXT:   saturate(f /*float*/);
// saturate-NEXT: Is migrated to:
// saturate-NEXT:   dpct::clamp<float>(f, 0.0f, 1.0f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=mul24 | FileCheck %s -check-prefix=mul24
// mul24: CUDA API:
// mul24-NEXT:   mul24(a /*int*/, b /*int*/);
// mul24-NEXT: Is migrated to:
// mul24-NEXT:   sycl::mul24(a, b);
