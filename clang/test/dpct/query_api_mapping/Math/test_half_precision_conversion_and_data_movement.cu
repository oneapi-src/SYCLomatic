/// Half Precision Conversion and Data Movement

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float22half2_rn | FileCheck %s -check-prefix=__FLOAT22HALF2_RN
// __FLOAT22HALF2_RN: CUDA API:
// __FLOAT22HALF2_RN-NEXT:   __float22half2_rn(f /*float2*/);
// __FLOAT22HALF2_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __FLOAT22HALF2_RN-NEXT:   sycl::half2(sycl::ext::intel::math::float2half_rn(f[0]), sycl::ext::intel::math::float2half_rn(f[1]));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2half | FileCheck %s -check-prefix=__FLOAT2HALF
// __FLOAT2HALF: CUDA API:
// __FLOAT2HALF-NEXT:   __float2half(f /*float*/);
// __FLOAT2HALF-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __FLOAT2HALF-NEXT:   sycl::ext::intel::math::float2half_rn(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2half2_rn | FileCheck %s -check-prefix=__FLOAT2HALF2_RN
// __FLOAT2HALF2_RN: CUDA API:
// __FLOAT2HALF2_RN-NEXT:   __float2half2_rn(f /*float*/);
// __FLOAT2HALF2_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __FLOAT2HALF2_RN-NEXT:   sycl::half2(sycl::ext::intel::math::float2half_rn(f));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2half_rd | FileCheck %s -check-prefix=__FLOAT2HALF_RD
// __FLOAT2HALF_RD: CUDA API:
// __FLOAT2HALF_RD-NEXT:   __float2half_rd(f /*float*/);
// __FLOAT2HALF_RD-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __FLOAT2HALF_RD-NEXT:   sycl::ext::intel::math::float2half_rd(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2half_rn | FileCheck %s -check-prefix=__FLOAT2HALF_RN
// __FLOAT2HALF_RN: CUDA API:
// __FLOAT2HALF_RN-NEXT:   __float2half_rn(f /*float*/);
// __FLOAT2HALF_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __FLOAT2HALF_RN-NEXT:   sycl::ext::intel::math::float2half_rn(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2half_ru | FileCheck %s -check-prefix=__FLOAT2HALF_RU
// __FLOAT2HALF_RU: CUDA API:
// __FLOAT2HALF_RU-NEXT:   __float2half_ru(f /*float*/);
// __FLOAT2HALF_RU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __FLOAT2HALF_RU-NEXT:   sycl::ext::intel::math::float2half_ru(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2half_rz | FileCheck %s -check-prefix=__FLOAT2HALF_RZ
// __FLOAT2HALF_RZ: CUDA API:
// __FLOAT2HALF_RZ-NEXT:   __float2half_rz(f /*float*/);
// __FLOAT2HALF_RZ-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __FLOAT2HALF_RZ-NEXT:   sycl::ext::intel::math::float2half_rz(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__floats2half2_rn | FileCheck %s -check-prefix=__FLOATS2HALF2_RN
// __FLOATS2HALF2_RN: CUDA API:
// __FLOATS2HALF2_RN-NEXT:   __floats2half2_rn(f1 /*float*/, f2 /*float*/);
// __FLOATS2HALF2_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __FLOATS2HALF2_RN-NEXT:   sycl::half2(sycl::ext::intel::math::float2half_rn(f1), sycl::ext::intel::math::float2half_rn(f2));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__half22float2 | FileCheck %s -check-prefix=__HALF22FLOAT2
// __HALF22FLOAT2: CUDA API:
// __HALF22FLOAT2-NEXT:   __half22float2(h /*__half2*/);
// __HALF22FLOAT2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HALF22FLOAT2-NEXT:   sycl::float2(sycl::ext::intel::math::half2float(h[0]), sycl::ext::intel::math::half2float(h[1]));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__half2float | FileCheck %s -check-prefix=__HALF2FLOAT
// __HALF2FLOAT: CUDA API:
// __HALF2FLOAT-NEXT:   __half2float(h /*__half*/);
// __HALF2FLOAT-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HALF2FLOAT-NEXT:   sycl::ext::intel::math::half2float(h);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__half2half2 | FileCheck %s -check-prefix=__HALF2HALF2
// __HALF2HALF2: CUDA API:
// __HALF2HALF2-NEXT:   __half2half2(h /*__half*/);
// __HALF2HALF2-NEXT: Is migrated to:
// __HALF2HALF2-NEXT:   sycl::half2(h);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__half2int_rd | FileCheck %s -check-prefix=__HALF2INT_RD
// __HALF2INT_RD: CUDA API:
// __HALF2INT_RD-NEXT:   __half2int_rd(h /*__half*/);
// __HALF2INT_RD-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HALF2INT_RD-NEXT:   sycl::ext::intel::math::half2int_rd(h);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__half2int_rn | FileCheck %s -check-prefix=__HALF2INT_RN
// __HALF2INT_RN: CUDA API:
// __HALF2INT_RN-NEXT:   __half2int_rn(h /*__half*/);
// __HALF2INT_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HALF2INT_RN-NEXT:   sycl::ext::intel::math::half2int_rn(h);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__half2int_ru | FileCheck %s -check-prefix=__HALF2INT_RU
// __HALF2INT_RU: CUDA API:
// __HALF2INT_RU-NEXT:   __half2int_ru(h /*__half*/);
// __HALF2INT_RU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HALF2INT_RU-NEXT:   sycl::ext::intel::math::half2int_ru(h);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__half2int_rz | FileCheck %s -check-prefix=__HALF2INT_RZ
// __HALF2INT_RZ: CUDA API:
// __HALF2INT_RZ-NEXT:   __half2int_rz(h /*__half*/);
// __HALF2INT_RZ-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HALF2INT_RZ-NEXT:   sycl::ext::intel::math::half2int_rz(h);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__half2ll_rd | FileCheck %s -check-prefix=__HALF2LL_RD
// __HALF2LL_RD: CUDA API:
// __HALF2LL_RD-NEXT:   __half2ll_rd(h /*__half*/);
// __HALF2LL_RD-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HALF2LL_RD-NEXT:   sycl::ext::intel::math::half2ll_rd(h);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__half2ll_rn | FileCheck %s -check-prefix=__HALF2LL_RN
// __HALF2LL_RN: CUDA API:
// __HALF2LL_RN-NEXT:   __half2ll_rn(h /*__half*/);
// __HALF2LL_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HALF2LL_RN-NEXT:   sycl::ext::intel::math::half2ll_rn(h);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__half2ll_ru | FileCheck %s -check-prefix=__HALF2LL_RU
// __HALF2LL_RU: CUDA API:
// __HALF2LL_RU-NEXT:   __half2ll_ru(h /*__half*/);
// __HALF2LL_RU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HALF2LL_RU-NEXT:   sycl::ext::intel::math::half2ll_ru(h);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__half2ll_rz | FileCheck %s -check-prefix=__HALF2LL_RZ
// __HALF2LL_RZ: CUDA API:
// __HALF2LL_RZ-NEXT:   __half2ll_rz(h /*__half*/);
// __HALF2LL_RZ-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HALF2LL_RZ-NEXT:   sycl::ext::intel::math::half2ll_rz(h);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__half2short_rd | FileCheck %s -check-prefix=__HALF2SHORT_RD
// __HALF2SHORT_RD: CUDA API:
// __HALF2SHORT_RD-NEXT:   __half2short_rd(h /*__half*/);
// __HALF2SHORT_RD-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HALF2SHORT_RD-NEXT:   sycl::ext::intel::math::half2short_rd(h);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__half2short_rn | FileCheck %s -check-prefix=__HALF2SHORT_RN
// __HALF2SHORT_RN: CUDA API:
// __HALF2SHORT_RN-NEXT:   __half2short_rn(h /*__half*/);
// __HALF2SHORT_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HALF2SHORT_RN-NEXT:   sycl::ext::intel::math::half2short_rn(h);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__half2short_ru | FileCheck %s -check-prefix=__HALF2SHORT_RU
// __HALF2SHORT_RU: CUDA API:
// __HALF2SHORT_RU-NEXT:   __half2short_ru(h /*__half*/);
// __HALF2SHORT_RU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HALF2SHORT_RU-NEXT:   sycl::ext::intel::math::half2short_ru(h);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__half2short_rz | FileCheck %s -check-prefix=__HALF2SHORT_RZ
// __HALF2SHORT_RZ: CUDA API:
// __HALF2SHORT_RZ-NEXT:   __half2short_rz(h /*__half*/);
// __HALF2SHORT_RZ-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HALF2SHORT_RZ-NEXT:   sycl::ext::intel::math::half2short_rz(h);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__half2uint_rd | FileCheck %s -check-prefix=__HALF2UINT_RD
// __HALF2UINT_RD: CUDA API:
// __HALF2UINT_RD-NEXT:   __half2uint_rd(h /*__half*/);
// __HALF2UINT_RD-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HALF2UINT_RD-NEXT:   sycl::ext::intel::math::half2uint_rd(h);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__half2uint_rn | FileCheck %s -check-prefix=__HALF2UINT_RN
// __HALF2UINT_RN: CUDA API:
// __HALF2UINT_RN-NEXT:   __half2uint_rn(h /*__half*/);
// __HALF2UINT_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HALF2UINT_RN-NEXT:   sycl::ext::intel::math::half2uint_rn(h);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__half2uint_ru | FileCheck %s -check-prefix=__HALF2UINT_RU
// __HALF2UINT_RU: CUDA API:
// __HALF2UINT_RU-NEXT:   __half2uint_ru(h /*__half*/);
// __HALF2UINT_RU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HALF2UINT_RU-NEXT:   sycl::ext::intel::math::half2uint_ru(h);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__half2uint_rz | FileCheck %s -check-prefix=__HALF2UINT_RZ
// __HALF2UINT_RZ: CUDA API:
// __HALF2UINT_RZ-NEXT:   __half2uint_rz(h /*__half*/);
// __HALF2UINT_RZ-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HALF2UINT_RZ-NEXT:   sycl::ext::intel::math::half2uint_rz(h);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__half2ull_rd | FileCheck %s -check-prefix=__HALF2ULL_RD
// __HALF2ULL_RD: CUDA API:
// __HALF2ULL_RD-NEXT:   __half2ull_rd(h /*__half*/);
// __HALF2ULL_RD-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HALF2ULL_RD-NEXT:   sycl::ext::intel::math::half2ull_rd(h);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__half2ull_rn | FileCheck %s -check-prefix=__HALF2ULL_RN
// __HALF2ULL_RN: CUDA API:
// __HALF2ULL_RN-NEXT:   __half2ull_rn(h /*__half*/);
// __HALF2ULL_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HALF2ULL_RN-NEXT:   sycl::ext::intel::math::half2ull_rn(h);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__half2ull_ru | FileCheck %s -check-prefix=__HALF2ULL_RU
// __HALF2ULL_RU: CUDA API:
// __HALF2ULL_RU-NEXT:   __half2ull_ru(h /*__half*/);
// __HALF2ULL_RU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HALF2ULL_RU-NEXT:   sycl::ext::intel::math::half2ull_ru(h);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__half2ull_rz | FileCheck %s -check-prefix=__HALF2ULL_RZ
// __HALF2ULL_RZ: CUDA API:
// __HALF2ULL_RZ-NEXT:   __half2ull_rz(h /*__half*/);
// __HALF2ULL_RZ-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HALF2ULL_RZ-NEXT:   sycl::ext::intel::math::half2ull_rz(h);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__half2ushort_rd | FileCheck %s -check-prefix=__HALF2USHORT_RD
// __HALF2USHORT_RD: CUDA API:
// __HALF2USHORT_RD-NEXT:   __half2ushort_rd(h /*__half*/);
// __HALF2USHORT_RD-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HALF2USHORT_RD-NEXT:   sycl::ext::intel::math::half2ushort_rd(h);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__half2ushort_rn | FileCheck %s -check-prefix=__HALF2USHORT_RN
// __HALF2USHORT_RN: CUDA API:
// __HALF2USHORT_RN-NEXT:   __half2ushort_rn(h /*__half*/);
// __HALF2USHORT_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HALF2USHORT_RN-NEXT:   sycl::ext::intel::math::half2ushort_rn(h);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__half2ushort_ru | FileCheck %s -check-prefix=__HALF2USHORT_RU
// __HALF2USHORT_RU: CUDA API:
// __HALF2USHORT_RU-NEXT:   __half2ushort_ru(h /*__half*/);
// __HALF2USHORT_RU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HALF2USHORT_RU-NEXT:   sycl::ext::intel::math::half2ushort_ru(h);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__half2ushort_rz | FileCheck %s -check-prefix=__HALF2USHORT_RZ
// __HALF2USHORT_RZ: CUDA API:
// __HALF2USHORT_RZ-NEXT:   __half2ushort_rz(h /*__half*/);
// __HALF2USHORT_RZ-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HALF2USHORT_RZ-NEXT:   sycl::ext::intel::math::half2ushort_rz(h);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__half_as_short | FileCheck %s -check-prefix=__HALF_AS_SHORT
// __HALF_AS_SHORT: CUDA API:
// __HALF_AS_SHORT-NEXT:   __half_as_short(h /*__half*/);
// __HALF_AS_SHORT-NEXT: Is migrated to:
// __HALF_AS_SHORT-NEXT:   sycl::bit_cast<short, sycl::half>(h);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__half_as_ushort | FileCheck %s -check-prefix=__HALF_AS_USHORT
// __HALF_AS_USHORT: CUDA API:
// __HALF_AS_USHORT-NEXT:   __half_as_ushort(h /*__half*/);
// __HALF_AS_USHORT-NEXT: Is migrated to:
// __HALF_AS_USHORT-NEXT:   sycl::bit_cast<unsigned short, sycl::half>(h);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__halves2half2 | FileCheck %s -check-prefix=__HALVES2HALF2
// __HALVES2HALF2: CUDA API:
// __HALVES2HALF2-NEXT:   __halves2half2(h1 /*__half*/, h2 /*__half*/);
// __HALVES2HALF2-NEXT: Is migrated to:
// __HALVES2HALF2-NEXT:   sycl::half2(h1, h2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__high2float | FileCheck %s -check-prefix=__HIGH2FLOAT
// __HIGH2FLOAT: CUDA API:
// __HIGH2FLOAT-NEXT:   __high2float(h /*__half2*/);
// __HIGH2FLOAT-NEXT: Is migrated to:
// __HIGH2FLOAT-NEXT:   h[1];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__high2half | FileCheck %s -check-prefix=__HIGH2HALF
// __HIGH2HALF: CUDA API:
// __HIGH2HALF-NEXT:   __high2half(h /*__half2*/);
// __HIGH2HALF-NEXT: Is migrated to:
// __HIGH2HALF-NEXT:   h[1];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__high2half2 | FileCheck %s -check-prefix=__HIGH2HALF2
// __HIGH2HALF2: CUDA API:
// __HIGH2HALF2-NEXT:   __high2half2(h /*__half2*/);
// __HIGH2HALF2-NEXT: Is migrated to:
// __HIGH2HALF2-NEXT:   sycl::half2(h[1]);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__highs2half2 | FileCheck %s -check-prefix=__HIGHS2HALF2
// __HIGHS2HALF2: CUDA API:
// __HIGHS2HALF2-NEXT:   __highs2half2(h1 /*__half2*/, h2 /*__half2*/);
// __HIGHS2HALF2-NEXT: Is migrated to:
// __HIGHS2HALF2-NEXT:   sycl::half2(h1[1], h2[1]);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__int2half_rd | FileCheck %s -check-prefix=__INT2HALF_RD
// __INT2HALF_RD: CUDA API:
// __INT2HALF_RD-NEXT:   __int2half_rd(i /*int*/);
// __INT2HALF_RD-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __INT2HALF_RD-NEXT:   sycl::ext::intel::math::int2half_rd(i);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__int2half_rn | FileCheck %s -check-prefix=__INT2HALF_RN
// __INT2HALF_RN: CUDA API:
// __INT2HALF_RN-NEXT:   __int2half_rn(i /*int*/);
// __INT2HALF_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __INT2HALF_RN-NEXT:   sycl::ext::intel::math::int2half_rn(i);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__int2half_ru | FileCheck %s -check-prefix=__INT2HALF_RU
// __INT2HALF_RU: CUDA API:
// __INT2HALF_RU-NEXT:   __int2half_ru(i /*int*/);
// __INT2HALF_RU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __INT2HALF_RU-NEXT:   sycl::ext::intel::math::int2half_ru(i);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__int2half_rz | FileCheck %s -check-prefix=__INT2HALF_RZ
// __INT2HALF_RZ: CUDA API:
// __INT2HALF_RZ-NEXT:   __int2half_rz(i /*int*/);
// __INT2HALF_RZ-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __INT2HALF_RZ-NEXT:   sycl::ext::intel::math::int2half_rz(i);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ll2half_rd | FileCheck %s -check-prefix=__LL2HALF_RD
// __LL2HALF_RD: CUDA API:
// __LL2HALF_RD-NEXT:   __ll2half_rd(ll /*long long*/);
// __LL2HALF_RD-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __LL2HALF_RD-NEXT:   sycl::ext::intel::math::ll2half_rd(ll);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ll2half_rn | FileCheck %s -check-prefix=__LL2HALF_RN
// __LL2HALF_RN: CUDA API:
// __LL2HALF_RN-NEXT:   __ll2half_rn(ll /*long long*/);
// __LL2HALF_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __LL2HALF_RN-NEXT:   sycl::ext::intel::math::ll2half_rn(ll);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ll2half_ru | FileCheck %s -check-prefix=__LL2HALF_RU
// __LL2HALF_RU: CUDA API:
// __LL2HALF_RU-NEXT:   __ll2half_ru(ll /*long long*/);
// __LL2HALF_RU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __LL2HALF_RU-NEXT:   sycl::ext::intel::math::ll2half_ru(ll);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ll2half_rz | FileCheck %s -check-prefix=__LL2HALF_RZ
// __LL2HALF_RZ: CUDA API:
// __LL2HALF_RZ-NEXT:   __ll2half_rz(ll /*long long*/);
// __LL2HALF_RZ-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __LL2HALF_RZ-NEXT:   sycl::ext::intel::math::ll2half_rz(ll);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__low2float | FileCheck %s -check-prefix=__LOW2FLOAT
// __LOW2FLOAT: CUDA API:
// __LOW2FLOAT-NEXT:   __low2float(h /*__half2*/);
// __LOW2FLOAT-NEXT: Is migrated to:
// __LOW2FLOAT-NEXT:   h[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__low2half | FileCheck %s -check-prefix=__LOW2HALF
// __LOW2HALF: CUDA API:
// __LOW2HALF-NEXT:   __low2half(h /*__half2*/);
// __LOW2HALF-NEXT: Is migrated to:
// __LOW2HALF-NEXT:   h[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__low2half2 | FileCheck %s -check-prefix=__LOW2HALF2
// __LOW2HALF2: CUDA API:
// __LOW2HALF2-NEXT:   __low2half2(h /*__half2*/);
// __LOW2HALF2-NEXT: Is migrated to:
// __LOW2HALF2-NEXT:   sycl::half2(h[0]);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__lowhigh2highlow | FileCheck %s -check-prefix=__LOWHIGH2HIGHLOW
// __LOWHIGH2HIGHLOW: CUDA API:
// __LOWHIGH2HIGHLOW-NEXT:   __lowhigh2highlow(h /*__half2*/);
// __LOWHIGH2HIGHLOW-NEXT: Is migrated to:
// __LOWHIGH2HIGHLOW-NEXT:   sycl::half2(h[1], h[0]);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__lows2half2 | FileCheck %s -check-prefix=__LOWS2HALF2
// __LOWS2HALF2: CUDA API:
// __LOWS2HALF2-NEXT:   __lows2half2(h1 /*__half2*/, h2 /*__half2*/);
// __LOWS2HALF2-NEXT: Is migrated to:
// __LOWS2HALF2-NEXT:   sycl::half2(h1[0], h2[0]);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__short2half_rd | FileCheck %s -check-prefix=__SHORT2HALF_RD
// __SHORT2HALF_RD: CUDA API:
// __SHORT2HALF_RD-NEXT:   __short2half_rd(s /*short*/);
// __SHORT2HALF_RD-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __SHORT2HALF_RD-NEXT:   sycl::ext::intel::math::short2half_rd(s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__short2half_rn | FileCheck %s -check-prefix=__SHORT2HALF_RN
// __SHORT2HALF_RN: CUDA API:
// __SHORT2HALF_RN-NEXT:   __short2half_rn(s /*short*/);
// __SHORT2HALF_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __SHORT2HALF_RN-NEXT:   sycl::ext::intel::math::short2half_rn(s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__short2half_ru | FileCheck %s -check-prefix=__SHORT2HALF_RU
// __SHORT2HALF_RU: CUDA API:
// __SHORT2HALF_RU-NEXT:   __short2half_ru(s /*short*/);
// __SHORT2HALF_RU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __SHORT2HALF_RU-NEXT:   sycl::ext::intel::math::short2half_ru(s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__short2half_rz | FileCheck %s -check-prefix=__SHORT2HALF_RZ
// __SHORT2HALF_RZ: CUDA API:
// __SHORT2HALF_RZ-NEXT:   __short2half_rz(s /*short*/);
// __SHORT2HALF_RZ-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __SHORT2HALF_RZ-NEXT:   sycl::ext::intel::math::short2half_rz(s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__short_as_half | FileCheck %s -check-prefix=__SHORT_AS_HALF
// __SHORT_AS_HALF: CUDA API:
// __SHORT_AS_HALF-NEXT:   __short_as_half(s /*short*/);
// __SHORT_AS_HALF-NEXT: Is migrated to:
// __SHORT_AS_HALF-NEXT:   sycl::bit_cast<sycl::half, short>(s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__uint2half_rd | FileCheck %s -check-prefix=__UINT2HALF_RD
// __UINT2HALF_RD: CUDA API:
// __UINT2HALF_RD-NEXT:   __uint2half_rd(u /*unsigned*/);
// __UINT2HALF_RD-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __UINT2HALF_RD-NEXT:   sycl::ext::intel::math::uint2half_rd(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__uint2half_rn | FileCheck %s -check-prefix=__UINT2HALF_RN
// __UINT2HALF_RN: CUDA API:
// __UINT2HALF_RN-NEXT:   __uint2half_rn(u /*unsigned*/);
// __UINT2HALF_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __UINT2HALF_RN-NEXT:   sycl::ext::intel::math::uint2half_rn(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__uint2half_ru | FileCheck %s -check-prefix=__UINT2HALF_RU
// __UINT2HALF_RU: CUDA API:
// __UINT2HALF_RU-NEXT:   __uint2half_ru(u /*unsigned*/);
// __UINT2HALF_RU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __UINT2HALF_RU-NEXT:   sycl::ext::intel::math::uint2half_ru(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__uint2half_rz | FileCheck %s -check-prefix=__UINT2HALF_RZ
// __UINT2HALF_RZ: CUDA API:
// __UINT2HALF_RZ-NEXT:   __uint2half_rz(u /*unsigned*/);
// __UINT2HALF_RZ-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __UINT2HALF_RZ-NEXT:   sycl::ext::intel::math::uint2half_rz(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ull2half_rd | FileCheck %s -check-prefix=__ULL2HALF_RD
// __ULL2HALF_RD: CUDA API:
// __ULL2HALF_RD-NEXT:   __ull2half_rd(u /*unsigned long long*/);
// __ULL2HALF_RD-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __ULL2HALF_RD-NEXT:   sycl::ext::intel::math::ull2half_rd(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ull2half_rn | FileCheck %s -check-prefix=__ULL2HALF_RN
// __ULL2HALF_RN: CUDA API:
// __ULL2HALF_RN-NEXT:   __ull2half_rn(u /*unsigned long long*/);
// __ULL2HALF_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __ULL2HALF_RN-NEXT:   sycl::ext::intel::math::ull2half_rn(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ull2half_ru | FileCheck %s -check-prefix=__ULL2HALF_RU
// __ULL2HALF_RU: CUDA API:
// __ULL2HALF_RU-NEXT:   __ull2half_ru(u /*unsigned long long*/);
// __ULL2HALF_RU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __ULL2HALF_RU-NEXT:   sycl::ext::intel::math::ull2half_ru(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ull2half_rz | FileCheck %s -check-prefix=__ULL2HALF_RZ
// __ULL2HALF_RZ: CUDA API:
// __ULL2HALF_RZ-NEXT:   __ull2half_rz(u /*unsigned long long*/);
// __ULL2HALF_RZ-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __ULL2HALF_RZ-NEXT:   sycl::ext::intel::math::ull2half_rz(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ushort2half_rd | FileCheck %s -check-prefix=__USHORT2HALF_RD
// __USHORT2HALF_RD: CUDA API:
// __USHORT2HALF_RD-NEXT:   __ushort2half_rd(u /*unsigned short*/);
// __USHORT2HALF_RD-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __USHORT2HALF_RD-NEXT:   sycl::ext::intel::math::ushort2half_rd(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ushort2half_rn | FileCheck %s -check-prefix=__USHORT2HALF_RN
// __USHORT2HALF_RN: CUDA API:
// __USHORT2HALF_RN-NEXT:   __ushort2half_rn(u /*unsigned short*/);
// __USHORT2HALF_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __USHORT2HALF_RN-NEXT:   sycl::ext::intel::math::ushort2half_rn(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ushort2half_ru | FileCheck %s -check-prefix=__USHORT2HALF_RU
// __USHORT2HALF_RU: CUDA API:
// __USHORT2HALF_RU-NEXT:   __ushort2half_ru(u /*unsigned short*/);
// __USHORT2HALF_RU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __USHORT2HALF_RU-NEXT:   sycl::ext::intel::math::ushort2half_ru(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ushort2half_rz | FileCheck %s -check-prefix=__USHORT2HALF_RZ
// __USHORT2HALF_RZ: CUDA API:
// __USHORT2HALF_RZ-NEXT:   __ushort2half_rz(u /*unsigned short*/);
// __USHORT2HALF_RZ-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __USHORT2HALF_RZ-NEXT:   sycl::ext::intel::math::ushort2half_rz(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ushort_as_half | FileCheck %s -check-prefix=__USHORT_AS_HALF
// __USHORT_AS_HALF: CUDA API:
// __USHORT_AS_HALF-NEXT:   __ushort_as_half(u /*unsigned short*/);
// __USHORT_AS_HALF-NEXT: Is migrated to:
// __USHORT_AS_HALF-NEXT:   sycl::bit_cast<sycl::half, unsigned short>(u);
