// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2

/// Bfloat16 Precision Conversion And Data Movement

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__bfloat1622float2 | FileCheck %s -check-prefix=__BFLOAT1622FLOAT2
// __BFLOAT1622FLOAT2: CUDA API:
// __BFLOAT1622FLOAT2-NEXT:   __bfloat1622float2(b /*__nv_bfloat162*/);
// __BFLOAT1622FLOAT2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __BFLOAT1622FLOAT2-NEXT:   sycl::float2(sycl::ext::intel::math::bfloat162float(b[0]), sycl::ext::intel::math::bfloat162float(b[1]));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__bfloat162bfloat162 | FileCheck %s -check-prefix=__BFLOAT162BFLOAT162
// __BFLOAT162BFLOAT162: CUDA API:
// __BFLOAT162BFLOAT162-NEXT:   __bfloat162bfloat162(b /*__nv_bfloat16*/);
// __BFLOAT162BFLOAT162-NEXT: Is migrated to:
// __BFLOAT162BFLOAT162-NEXT:   sycl::marray<sycl::ext::oneapi::bfloat16, 2>(b, b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__bfloat162float | FileCheck %s -check-prefix=__BFLOAT162FLOAT
// __BFLOAT162FLOAT: CUDA API:
// __BFLOAT162FLOAT-NEXT:   __bfloat162float(b /*__nv_bfloat16*/);
// __BFLOAT162FLOAT-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __BFLOAT162FLOAT-NEXT:   sycl::ext::intel::math::bfloat162float(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__bfloat162int_rd | FileCheck %s -check-prefix=__BFLOAT162INT_RD
// __BFLOAT162INT_RD: CUDA API:
// __BFLOAT162INT_RD-NEXT:   __bfloat162int_rd(b /*__nv_bfloat16*/);
// __BFLOAT162INT_RD-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __BFLOAT162INT_RD-NEXT:   sycl::ext::intel::math::bfloat162int_rd(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__bfloat162int_rn | FileCheck %s -check-prefix=__BFLOAT162INT_RN
// __BFLOAT162INT_RN: CUDA API:
// __BFLOAT162INT_RN-NEXT:   __bfloat162int_rn(b /*__nv_bfloat16*/);
// __BFLOAT162INT_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __BFLOAT162INT_RN-NEXT:   sycl::ext::intel::math::bfloat162int_rn(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__bfloat162int_ru | FileCheck %s -check-prefix=__BFLOAT162INT_RU
// __BFLOAT162INT_RU: CUDA API:
// __BFLOAT162INT_RU-NEXT:   __bfloat162int_ru(b /*__nv_bfloat16*/);
// __BFLOAT162INT_RU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __BFLOAT162INT_RU-NEXT:   sycl::ext::intel::math::bfloat162int_ru(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__bfloat162int_rz | FileCheck %s -check-prefix=__BFLOAT162INT_RZ
// __BFLOAT162INT_RZ: CUDA API:
// __BFLOAT162INT_RZ-NEXT:   __bfloat162int_rz(b /*__nv_bfloat16*/);
// __BFLOAT162INT_RZ-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __BFLOAT162INT_RZ-NEXT:   sycl::ext::intel::math::bfloat162int_rz(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__bfloat162ll_rd | FileCheck %s -check-prefix=__BFLOAT162LL_RD
// __BFLOAT162LL_RD: CUDA API:
// __BFLOAT162LL_RD-NEXT:   __bfloat162ll_rd(b /*__nv_bfloat16*/);
// __BFLOAT162LL_RD-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __BFLOAT162LL_RD-NEXT:   sycl::ext::intel::math::bfloat162ll_rd(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__bfloat162ll_rn | FileCheck %s -check-prefix=__BFLOAT162LL_RN
// __BFLOAT162LL_RN: CUDA API:
// __BFLOAT162LL_RN-NEXT:   __bfloat162ll_rn(b /*__nv_bfloat16*/);
// __BFLOAT162LL_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __BFLOAT162LL_RN-NEXT:   sycl::ext::intel::math::bfloat162ll_rn(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__bfloat162ll_ru | FileCheck %s -check-prefix=__BFLOAT162LL_RU
// __BFLOAT162LL_RU: CUDA API:
// __BFLOAT162LL_RU-NEXT:   __bfloat162ll_ru(b /*__nv_bfloat16*/);
// __BFLOAT162LL_RU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __BFLOAT162LL_RU-NEXT:   sycl::ext::intel::math::bfloat162ll_ru(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__bfloat162ll_rz | FileCheck %s -check-prefix=__BFLOAT162LL_RZ
// __BFLOAT162LL_RZ: CUDA API:
// __BFLOAT162LL_RZ-NEXT:   __bfloat162ll_rz(b /*__nv_bfloat16*/);
// __BFLOAT162LL_RZ-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __BFLOAT162LL_RZ-NEXT:   sycl::ext::intel::math::bfloat162ll_rz(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__bfloat162short_rd | FileCheck %s -check-prefix=__BFLOAT162SHORT_RD
// __BFLOAT162SHORT_RD: CUDA API:
// __BFLOAT162SHORT_RD-NEXT:   __bfloat162short_rd(b /*__nv_bfloat16*/);
// __BFLOAT162SHORT_RD-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __BFLOAT162SHORT_RD-NEXT:   sycl::ext::intel::math::bfloat162short_rd(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__bfloat162short_rn | FileCheck %s -check-prefix=__BFLOAT162SHORT_RN
// __BFLOAT162SHORT_RN: CUDA API:
// __BFLOAT162SHORT_RN-NEXT:   __bfloat162short_rn(b /*__nv_bfloat16*/);
// __BFLOAT162SHORT_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __BFLOAT162SHORT_RN-NEXT:   sycl::ext::intel::math::bfloat162short_rn(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__bfloat162short_ru | FileCheck %s -check-prefix=__BFLOAT162SHORT_RU
// __BFLOAT162SHORT_RU: CUDA API:
// __BFLOAT162SHORT_RU-NEXT:   __bfloat162short_ru(b /*__nv_bfloat16*/);
// __BFLOAT162SHORT_RU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __BFLOAT162SHORT_RU-NEXT:   sycl::ext::intel::math::bfloat162short_ru(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__bfloat162short_rz | FileCheck %s -check-prefix=__BFLOAT162SHORT_RZ
// __BFLOAT162SHORT_RZ: CUDA API:
// __BFLOAT162SHORT_RZ-NEXT:   __bfloat162short_rz(b /*__nv_bfloat16*/);
// __BFLOAT162SHORT_RZ-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __BFLOAT162SHORT_RZ-NEXT:   sycl::ext::intel::math::bfloat162short_rz(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__bfloat162uint_rd | FileCheck %s -check-prefix=__BFLOAT162UINT_RD
// __BFLOAT162UINT_RD: CUDA API:
// __BFLOAT162UINT_RD-NEXT:   __bfloat162uint_rd(b /*__nv_bfloat16*/);
// __BFLOAT162UINT_RD-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __BFLOAT162UINT_RD-NEXT:   sycl::ext::intel::math::bfloat162uint_rd(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__bfloat162uint_rn | FileCheck %s -check-prefix=__BFLOAT162UINT_RN
// __BFLOAT162UINT_RN: CUDA API:
// __BFLOAT162UINT_RN-NEXT:   __bfloat162uint_rn(b /*__nv_bfloat16*/);
// __BFLOAT162UINT_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __BFLOAT162UINT_RN-NEXT:   sycl::ext::intel::math::bfloat162uint_rn(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__bfloat162uint_ru | FileCheck %s -check-prefix=__BFLOAT162UINT_RU
// __BFLOAT162UINT_RU: CUDA API:
// __BFLOAT162UINT_RU-NEXT:   __bfloat162uint_ru(b /*__nv_bfloat16*/);
// __BFLOAT162UINT_RU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __BFLOAT162UINT_RU-NEXT:   sycl::ext::intel::math::bfloat162uint_ru(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__bfloat162uint_rz | FileCheck %s -check-prefix=__BFLOAT162UINT_RZ
// __BFLOAT162UINT_RZ: CUDA API:
// __BFLOAT162UINT_RZ-NEXT:   __bfloat162uint_rz(b /*__nv_bfloat16*/);
// __BFLOAT162UINT_RZ-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __BFLOAT162UINT_RZ-NEXT:   sycl::ext::intel::math::bfloat162uint_rz(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__bfloat162ull_rd | FileCheck %s -check-prefix=__BFLOAT162ULL_RD
// __BFLOAT162ULL_RD: CUDA API:
// __BFLOAT162ULL_RD-NEXT:   __bfloat162ull_rd(b /*__nv_bfloat16*/);
// __BFLOAT162ULL_RD-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __BFLOAT162ULL_RD-NEXT:   sycl::ext::intel::math::bfloat162ull_rd(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__bfloat162ull_rn | FileCheck %s -check-prefix=__BFLOAT162ULL_RN
// __BFLOAT162ULL_RN: CUDA API:
// __BFLOAT162ULL_RN-NEXT:   __bfloat162ull_rn(b /*__nv_bfloat16*/);
// __BFLOAT162ULL_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __BFLOAT162ULL_RN-NEXT:   sycl::ext::intel::math::bfloat162ull_rn(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__bfloat162ull_ru | FileCheck %s -check-prefix=__BFLOAT162ULL_RU
// __BFLOAT162ULL_RU: CUDA API:
// __BFLOAT162ULL_RU-NEXT:   __bfloat162ull_ru(b /*__nv_bfloat16*/);
// __BFLOAT162ULL_RU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __BFLOAT162ULL_RU-NEXT:   sycl::ext::intel::math::bfloat162ull_ru(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__bfloat162ull_rz | FileCheck %s -check-prefix=__BFLOAT162ULL_RZ
// __BFLOAT162ULL_RZ: CUDA API:
// __BFLOAT162ULL_RZ-NEXT:   __bfloat162ull_rz(b /*__nv_bfloat16*/);
// __BFLOAT162ULL_RZ-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __BFLOAT162ULL_RZ-NEXT:   sycl::ext::intel::math::bfloat162ull_rz(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__bfloat162ushort_rd | FileCheck %s -check-prefix=__BFLOAT162USHORT_RD
// __BFLOAT162USHORT_RD: CUDA API:
// __BFLOAT162USHORT_RD-NEXT:   __bfloat162ushort_rd(b /*__nv_bfloat16*/);
// __BFLOAT162USHORT_RD-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __BFLOAT162USHORT_RD-NEXT:   sycl::ext::intel::math::bfloat162ushort_rd(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__bfloat162ushort_rn | FileCheck %s -check-prefix=__BFLOAT162USHORT_RN
// __BFLOAT162USHORT_RN: CUDA API:
// __BFLOAT162USHORT_RN-NEXT:   __bfloat162ushort_rn(b /*__nv_bfloat16*/);
// __BFLOAT162USHORT_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __BFLOAT162USHORT_RN-NEXT:   sycl::ext::intel::math::bfloat162ushort_rn(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__bfloat162ushort_ru | FileCheck %s -check-prefix=__BFLOAT162USHORT_RU
// __BFLOAT162USHORT_RU: CUDA API:
// __BFLOAT162USHORT_RU-NEXT:   __bfloat162ushort_ru(b /*__nv_bfloat16*/);
// __BFLOAT162USHORT_RU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __BFLOAT162USHORT_RU-NEXT:   sycl::ext::intel::math::bfloat162ushort_ru(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__bfloat162ushort_rz | FileCheck %s -check-prefix=__BFLOAT162USHORT_RZ
// __BFLOAT162USHORT_RZ: CUDA API:
// __BFLOAT162USHORT_RZ-NEXT:   __bfloat162ushort_rz(b /*__nv_bfloat16*/);
// __BFLOAT162USHORT_RZ-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __BFLOAT162USHORT_RZ-NEXT:   sycl::ext::intel::math::bfloat162ushort_rz(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__bfloat16_as_short | FileCheck %s -check-prefix=__BFLOAT16_AS_SHORT
// __BFLOAT16_AS_SHORT: CUDA API:
// __BFLOAT16_AS_SHORT-NEXT:   __bfloat16_as_short(b /*__nv_bfloat16*/);
// __BFLOAT16_AS_SHORT-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __BFLOAT16_AS_SHORT-NEXT:   sycl::ext::intel::math::bfloat16_as_short(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__bfloat16_as_ushort | FileCheck %s -check-prefix=__BFLOAT16_AS_USHORT
// __BFLOAT16_AS_USHORT: CUDA API:
// __BFLOAT16_AS_USHORT-NEXT:   __bfloat16_as_ushort(b /*__nv_bfloat16*/);
// __BFLOAT16_AS_USHORT-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __BFLOAT16_AS_USHORT-NEXT:   sycl::ext::intel::math::bfloat16_as_ushort(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2bfloat16 | FileCheck %s -check-prefix=__DOUBLE2BFLOAT16
// __DOUBLE2BFLOAT16: CUDA API:
// __DOUBLE2BFLOAT16-NEXT:   __double2bfloat16(d /*double*/);
// __DOUBLE2BFLOAT16-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __DOUBLE2BFLOAT16-NEXT:   sycl::ext::intel::math::double2bfloat16(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float22bfloat162_rn | FileCheck %s -check-prefix=__FLOAT22BFLOAT162_RN
// __FLOAT22BFLOAT162_RN: CUDA API:
// __FLOAT22BFLOAT162_RN-NEXT:   __float22bfloat162_rn(f /*float2*/);
// __FLOAT22BFLOAT162_RN-NEXT: Is migrated to:
// __FLOAT22BFLOAT162_RN-NEXT:   sycl::marray<sycl::ext::oneapi::bfloat16, 2>(f[0], f[1]);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2bfloat16 | FileCheck %s -check-prefix=__FLOAT2BFLOAT16
// __FLOAT2BFLOAT16: CUDA API:
// __FLOAT2BFLOAT16-NEXT:   __float2bfloat16(f /*float*/);
// __FLOAT2BFLOAT16-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __FLOAT2BFLOAT16-NEXT:   sycl::ext::intel::math::float2bfloat16(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2bfloat162_rn | FileCheck %s -check-prefix=__FLOAT2BFLOAT162_RN
// __FLOAT2BFLOAT162_RN: CUDA API:
// __FLOAT2BFLOAT162_RN-NEXT:   __float2bfloat162_rn(f /*float*/);
// __FLOAT2BFLOAT162_RN-NEXT: Is migrated to:
// __FLOAT2BFLOAT162_RN-NEXT:   sycl::marray<sycl::ext::oneapi::bfloat16, 2>(f, f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2bfloat16_rd | FileCheck %s -check-prefix=__FLOAT2BFLOAT16_RD
// __FLOAT2BFLOAT16_RD: CUDA API:
// __FLOAT2BFLOAT16_RD-NEXT:   __float2bfloat16_rd(f /*float*/);
// __FLOAT2BFLOAT16_RD-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __FLOAT2BFLOAT16_RD-NEXT:   sycl::ext::intel::math::float2bfloat16_rd(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2bfloat16_rn | FileCheck %s -check-prefix=__FLOAT2BFLOAT16_RN
// __FLOAT2BFLOAT16_RN: CUDA API:
// __FLOAT2BFLOAT16_RN-NEXT:   __float2bfloat16_rn(f /*float*/);
// __FLOAT2BFLOAT16_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __FLOAT2BFLOAT16_RN-NEXT:   sycl::ext::intel::math::float2bfloat16_rn(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2bfloat16_ru | FileCheck %s -check-prefix=__FLOAT2BFLOAT16_RU
// __FLOAT2BFLOAT16_RU: CUDA API:
// __FLOAT2BFLOAT16_RU-NEXT:   __float2bfloat16_ru(f /*float*/);
// __FLOAT2BFLOAT16_RU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __FLOAT2BFLOAT16_RU-NEXT:   sycl::ext::intel::math::float2bfloat16_ru(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2bfloat16_rz | FileCheck %s -check-prefix=__FLOAT2BFLOAT16_RZ
// __FLOAT2BFLOAT16_RZ: CUDA API:
// __FLOAT2BFLOAT16_RZ-NEXT:   __float2bfloat16_rz(f /*float*/);
// __FLOAT2BFLOAT16_RZ-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __FLOAT2BFLOAT16_RZ-NEXT:   sycl::ext::intel::math::float2bfloat16_rz(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__floats2bfloat162_rn | FileCheck %s -check-prefix=__FLOATS2BFLOAT162_RN
// __FLOATS2BFLOAT162_RN: CUDA API:
// __FLOATS2BFLOAT162_RN-NEXT:   __floats2bfloat162_rn(f1 /*float*/, f2 /*float*/);
// __FLOATS2BFLOAT162_RN-NEXT: Is migrated to:
// __FLOATS2BFLOAT162_RN-NEXT:   sycl::marray<sycl::ext::oneapi::bfloat16, 2>(f1, f2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__halves2bfloat162 | FileCheck %s -check-prefix=__HALVES2BFLOAT162
// __HALVES2BFLOAT162: CUDA API:
// __HALVES2BFLOAT162-NEXT:   __halves2bfloat162(b1 /*__nv_bfloat16*/, b2 /*__nv_bfloat16*/);
// __HALVES2BFLOAT162-NEXT: Is migrated to:
// __HALVES2BFLOAT162-NEXT:   sycl::marray<sycl::ext::oneapi::bfloat16, 2>(b1, b2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__high2bfloat16 | FileCheck %s -check-prefix=__HIGH2BFLOAT16
// __HIGH2BFLOAT16: CUDA API:
// __HIGH2BFLOAT16-NEXT:   __high2bfloat16(b /*__nv_bfloat162*/);
// __HIGH2BFLOAT16-NEXT: Is migrated to:
// __HIGH2BFLOAT16-NEXT:   sycl::ext::oneapi::bfloat16(b[1]);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__high2bfloat162 | FileCheck %s -check-prefix=__HIGH2BFLOAT162
// __HIGH2BFLOAT162: CUDA API:
// __HIGH2BFLOAT162-NEXT:   __high2bfloat162(b /*__nv_bfloat162*/);
// __HIGH2BFLOAT162-NEXT: Is migrated to:
// __HIGH2BFLOAT162-NEXT:   sycl::marray<sycl::ext::oneapi::bfloat16, 2>(b[1], b[1]);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__highs2bfloat162 | FileCheck %s -check-prefix=__HIGHS2BFLOAT162
// __HIGHS2BFLOAT162: CUDA API:
// __HIGHS2BFLOAT162-NEXT:   __highs2bfloat162(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HIGHS2BFLOAT162-NEXT: Is migrated to:
// __HIGHS2BFLOAT162-NEXT:   sycl::marray<sycl::ext::oneapi::bfloat16, 2>(b1[1], b2[1]);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__int2bfloat16_rd | FileCheck %s -check-prefix=__INT2BFLOAT16_RD
// __INT2BFLOAT16_RD: CUDA API:
// __INT2BFLOAT16_RD-NEXT:   __int2bfloat16_rd(i /*int*/);
// __INT2BFLOAT16_RD-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __INT2BFLOAT16_RD-NEXT:   sycl::ext::intel::math::int2bfloat16_rd(i);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__int2bfloat16_rn | FileCheck %s -check-prefix=__INT2BFLOAT16_RN
// __INT2BFLOAT16_RN: CUDA API:
// __INT2BFLOAT16_RN-NEXT:   __int2bfloat16_rn(i /*int*/);
// __INT2BFLOAT16_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __INT2BFLOAT16_RN-NEXT:   sycl::ext::intel::math::int2bfloat16_rn(i);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__int2bfloat16_ru | FileCheck %s -check-prefix=__INT2BFLOAT16_RU
// __INT2BFLOAT16_RU: CUDA API:
// __INT2BFLOAT16_RU-NEXT:   __int2bfloat16_ru(i /*int*/);
// __INT2BFLOAT16_RU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __INT2BFLOAT16_RU-NEXT:   sycl::ext::intel::math::int2bfloat16_ru(i);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__int2bfloat16_rz | FileCheck %s -check-prefix=__INT2BFLOAT16_RZ
// __INT2BFLOAT16_RZ: CUDA API:
// __INT2BFLOAT16_RZ-NEXT:   __int2bfloat16_rz(i /*int*/);
// __INT2BFLOAT16_RZ-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __INT2BFLOAT16_RZ-NEXT:   sycl::ext::intel::math::int2bfloat16_rz(i);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ll2bfloat16_rd | FileCheck %s -check-prefix=__LL2BFLOAT16_RD
// __LL2BFLOAT16_RD: CUDA API:
// __LL2BFLOAT16_RD-NEXT:   __ll2bfloat16_rd(ll /*long long*/);
// __LL2BFLOAT16_RD-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __LL2BFLOAT16_RD-NEXT:   sycl::ext::intel::math::ll2bfloat16_rd(ll);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ll2bfloat16_rn | FileCheck %s -check-prefix=__LL2BFLOAT16_RN
// __LL2BFLOAT16_RN: CUDA API:
// __LL2BFLOAT16_RN-NEXT:   __ll2bfloat16_rn(ll /*long long*/);
// __LL2BFLOAT16_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __LL2BFLOAT16_RN-NEXT:   sycl::ext::intel::math::ll2bfloat16_rn(ll);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ll2bfloat16_ru | FileCheck %s -check-prefix=__LL2BFLOAT16_RU
// __LL2BFLOAT16_RU: CUDA API:
// __LL2BFLOAT16_RU-NEXT:   __ll2bfloat16_ru(ll /*long long*/);
// __LL2BFLOAT16_RU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __LL2BFLOAT16_RU-NEXT:   sycl::ext::intel::math::ll2bfloat16_ru(ll);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ll2bfloat16_rz | FileCheck %s -check-prefix=__LL2BFLOAT16_RZ
// __LL2BFLOAT16_RZ: CUDA API:
// __LL2BFLOAT16_RZ-NEXT:   __ll2bfloat16_rz(ll /*long long*/);
// __LL2BFLOAT16_RZ-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __LL2BFLOAT16_RZ-NEXT:   sycl::ext::intel::math::ll2bfloat16_rz(ll);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__low2bfloat16 | FileCheck %s -check-prefix=__LOW2BFLOAT16
// __LOW2BFLOAT16: CUDA API:
// __LOW2BFLOAT16-NEXT:   __low2bfloat16(b /*__nv_bfloat162*/);
// __LOW2BFLOAT16-NEXT: Is migrated to:
// __LOW2BFLOAT16-NEXT:   sycl::ext::oneapi::bfloat16(b[0]);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__low2bfloat162 | FileCheck %s -check-prefix=__LOW2BFLOAT162
// __LOW2BFLOAT162: CUDA API:
// __LOW2BFLOAT162-NEXT:   __low2bfloat162(b /*__nv_bfloat162*/);
// __LOW2BFLOAT162-NEXT: Is migrated to:
// __LOW2BFLOAT162-NEXT:   sycl::marray<sycl::ext::oneapi::bfloat16, 2>(b[0], b[0]);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__lows2bfloat162 | FileCheck %s -check-prefix=__LOWS2BFLOAT162
// __LOWS2BFLOAT162: CUDA API:
// __LOWS2BFLOAT162-NEXT:   __lows2bfloat162(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __LOWS2BFLOAT162-NEXT: Is migrated to:
// __LOWS2BFLOAT162-NEXT:   sycl::marray<sycl::ext::oneapi::bfloat16, 2>(b1[0], b2[0]);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__short2bfloat16_rd | FileCheck %s -check-prefix=__SHORT2BFLOAT16_RD
// __SHORT2BFLOAT16_RD: CUDA API:
// __SHORT2BFLOAT16_RD-NEXT:   __short2bfloat16_rd(s /*short*/);
// __SHORT2BFLOAT16_RD-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __SHORT2BFLOAT16_RD-NEXT:   sycl::ext::intel::math::short2bfloat16_rd(s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__short2bfloat16_rn | FileCheck %s -check-prefix=__SHORT2BFLOAT16_RN
// __SHORT2BFLOAT16_RN: CUDA API:
// __SHORT2BFLOAT16_RN-NEXT:   __short2bfloat16_rn(s /*short*/);
// __SHORT2BFLOAT16_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __SHORT2BFLOAT16_RN-NEXT:   sycl::ext::intel::math::short2bfloat16_rn(s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__short2bfloat16_ru | FileCheck %s -check-prefix=__SHORT2BFLOAT16_RU
// __SHORT2BFLOAT16_RU: CUDA API:
// __SHORT2BFLOAT16_RU-NEXT:   __short2bfloat16_ru(s /*short*/);
// __SHORT2BFLOAT16_RU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __SHORT2BFLOAT16_RU-NEXT:   sycl::ext::intel::math::short2bfloat16_ru(s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__short2bfloat16_rz | FileCheck %s -check-prefix=__SHORT2BFLOAT16_RZ
// __SHORT2BFLOAT16_RZ: CUDA API:
// __SHORT2BFLOAT16_RZ-NEXT:   __short2bfloat16_rz(s /*short*/);
// __SHORT2BFLOAT16_RZ-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __SHORT2BFLOAT16_RZ-NEXT:   sycl::ext::intel::math::short2bfloat16_rz(s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__short_as_bfloat16 | FileCheck %s -check-prefix=__SHORT_AS_BFLOAT16
// __SHORT_AS_BFLOAT16: CUDA API:
// __SHORT_AS_BFLOAT16-NEXT:   __short_as_bfloat16(s /*short*/);
// __SHORT_AS_BFLOAT16-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __SHORT_AS_BFLOAT16-NEXT:   sycl::ext::intel::math::short_as_bfloat16(s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__uint2bfloat16_rd | FileCheck %s -check-prefix=__UINT2BFLOAT16_RD
// __UINT2BFLOAT16_RD: CUDA API:
// __UINT2BFLOAT16_RD-NEXT:   __uint2bfloat16_rd(u /*unsigned*/);
// __UINT2BFLOAT16_RD-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __UINT2BFLOAT16_RD-NEXT:   sycl::ext::intel::math::uint2bfloat16_rd(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__uint2bfloat16_rn | FileCheck %s -check-prefix=__UINT2BFLOAT16_RN
// __UINT2BFLOAT16_RN: CUDA API:
// __UINT2BFLOAT16_RN-NEXT:   __uint2bfloat16_rn(u /*unsigned*/);
// __UINT2BFLOAT16_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __UINT2BFLOAT16_RN-NEXT:   sycl::ext::intel::math::uint2bfloat16_rn(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__uint2bfloat16_ru | FileCheck %s -check-prefix=__UINT2BFLOAT16_RU
// __UINT2BFLOAT16_RU: CUDA API:
// __UINT2BFLOAT16_RU-NEXT:   __uint2bfloat16_ru(u /*unsigned*/);
// __UINT2BFLOAT16_RU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __UINT2BFLOAT16_RU-NEXT:   sycl::ext::intel::math::uint2bfloat16_ru(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__uint2bfloat16_rz | FileCheck %s -check-prefix=__UINT2BFLOAT16_RZ
// __UINT2BFLOAT16_RZ: CUDA API:
// __UINT2BFLOAT16_RZ-NEXT:   __uint2bfloat16_rz(u /*unsigned*/);
// __UINT2BFLOAT16_RZ-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __UINT2BFLOAT16_RZ-NEXT:   sycl::ext::intel::math::uint2bfloat16_rz(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ull2bfloat16_rd | FileCheck %s -check-prefix=__ULL2BFLOAT16_RD
// __ULL2BFLOAT16_RD: CUDA API:
// __ULL2BFLOAT16_RD-NEXT:   __ull2bfloat16_rd(u /*unsigned long long*/);
// __ULL2BFLOAT16_RD-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __ULL2BFLOAT16_RD-NEXT:   sycl::ext::intel::math::ull2bfloat16_rd(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ull2bfloat16_rn | FileCheck %s -check-prefix=__ULL2BFLOAT16_RN
// __ULL2BFLOAT16_RN: CUDA API:
// __ULL2BFLOAT16_RN-NEXT:   __ull2bfloat16_rn(u /*unsigned long long*/);
// __ULL2BFLOAT16_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __ULL2BFLOAT16_RN-NEXT:   sycl::ext::intel::math::ull2bfloat16_rn(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ull2bfloat16_ru | FileCheck %s -check-prefix=__ULL2BFLOAT16_RU
// __ULL2BFLOAT16_RU: CUDA API:
// __ULL2BFLOAT16_RU-NEXT:   __ull2bfloat16_ru(u /*unsigned long long*/);
// __ULL2BFLOAT16_RU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __ULL2BFLOAT16_RU-NEXT:   sycl::ext::intel::math::ull2bfloat16_ru(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ull2bfloat16_rz | FileCheck %s -check-prefix=__ULL2BFLOAT16_RZ
// __ULL2BFLOAT16_RZ: CUDA API:
// __ULL2BFLOAT16_RZ-NEXT:   __ull2bfloat16_rz(u /*unsigned long long*/);
// __ULL2BFLOAT16_RZ-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __ULL2BFLOAT16_RZ-NEXT:   sycl::ext::intel::math::ull2bfloat16_rz(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ushort2bfloat16_rd | FileCheck %s -check-prefix=__USHORT2BFLOAT16_RD
// __USHORT2BFLOAT16_RD: CUDA API:
// __USHORT2BFLOAT16_RD-NEXT:   __ushort2bfloat16_rd(u /*unsigned short*/);
// __USHORT2BFLOAT16_RD-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __USHORT2BFLOAT16_RD-NEXT:   sycl::ext::intel::math::ushort2bfloat16_rd(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ushort2bfloat16_rn | FileCheck %s -check-prefix=__USHORT2BFLOAT16_RN
// __USHORT2BFLOAT16_RN: CUDA API:
// __USHORT2BFLOAT16_RN-NEXT:   __ushort2bfloat16_rn(u /*unsigned short*/);
// __USHORT2BFLOAT16_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __USHORT2BFLOAT16_RN-NEXT:   sycl::ext::intel::math::ushort2bfloat16_rn(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ushort2bfloat16_ru | FileCheck %s -check-prefix=__USHORT2BFLOAT16_RU
// __USHORT2BFLOAT16_RU: CUDA API:
// __USHORT2BFLOAT16_RU-NEXT:   __ushort2bfloat16_ru(u /*unsigned short*/);
// __USHORT2BFLOAT16_RU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __USHORT2BFLOAT16_RU-NEXT:   sycl::ext::intel::math::ushort2bfloat16_ru(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ushort2bfloat16_rz | FileCheck %s -check-prefix=__USHORT2BFLOAT16_RZ
// __USHORT2BFLOAT16_RZ: CUDA API:
// __USHORT2BFLOAT16_RZ-NEXT:   __ushort2bfloat16_rz(u /*unsigned short*/);
// __USHORT2BFLOAT16_RZ-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __USHORT2BFLOAT16_RZ-NEXT:   sycl::ext::intel::math::ushort2bfloat16_rz(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ushort_as_bfloat16 | FileCheck %s -check-prefix=__USHORT_AS_BFLOAT16
// __USHORT_AS_BFLOAT16: CUDA API:
// __USHORT_AS_BFLOAT16-NEXT:   __ushort_as_bfloat16(u /*unsigned short*/);
// __USHORT_AS_BFLOAT16-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __USHORT_AS_BFLOAT16-NEXT:   sycl::ext::intel::math::ushort_as_bfloat16(u);
