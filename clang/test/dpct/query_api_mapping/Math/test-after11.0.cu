// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_bfloat162 | FileCheck %s -check-prefix=make_bfloat162
// make_bfloat162: CUDA API:
// make_bfloat162-NEXT:   make_bfloat162(bf1 /*__nv_bfloat16*/, bf2 /*__nv_bfloat16*/);
// make_bfloat162-NEXT: Is migrated to:
// make_bfloat162-NEXT:   sycl::vec<sycl::ext::oneapi::bfloat16, 2>(bf1, bf2);
