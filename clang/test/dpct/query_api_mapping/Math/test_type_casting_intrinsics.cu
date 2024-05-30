/// Type Casting Intrinsics

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2float_rd | FileCheck %s -check-prefix=__DOUBLE2FLOAT_RD
// __DOUBLE2FLOAT_RD: CUDA API:
// __DOUBLE2FLOAT_RD-NEXT:   __double2float_rd(d /*double*/);
// __DOUBLE2FLOAT_RD-NEXT: Is migrated to:
// __DOUBLE2FLOAT_RD-NEXT:   sycl::vec<double, 1>{d}.convert<float, sycl::rounding_mode::rtn>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2float_rn | FileCheck %s -check-prefix=__DOUBLE2FLOAT_RN
// __DOUBLE2FLOAT_RN: CUDA API:
// __DOUBLE2FLOAT_RN-NEXT:   __double2float_rn(d /*double*/);
// __DOUBLE2FLOAT_RN-NEXT: Is migrated to:
// __DOUBLE2FLOAT_RN-NEXT:   sycl::vec<double, 1>{d}.convert<float, sycl::rounding_mode::rte>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2float_ru | FileCheck %s -check-prefix=__DOUBLE2FLOAT_RU
// __DOUBLE2FLOAT_RU: CUDA API:
// __DOUBLE2FLOAT_RU-NEXT:   __double2float_ru(d /*double*/);
// __DOUBLE2FLOAT_RU-NEXT: Is migrated to:
// __DOUBLE2FLOAT_RU-NEXT:   sycl::vec<double, 1>{d}.convert<float, sycl::rounding_mode::rtp>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2float_rz | FileCheck %s -check-prefix=__DOUBLE2FLOAT_RZ
// __DOUBLE2FLOAT_RZ: CUDA API:
// __DOUBLE2FLOAT_RZ-NEXT:   __double2float_rz(d /*double*/);
// __DOUBLE2FLOAT_RZ-NEXT: Is migrated to:
// __DOUBLE2FLOAT_RZ-NEXT:   sycl::vec<double, 1>{d}.convert<float, sycl::rounding_mode::rtz>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2hiint | FileCheck %s -check-prefix=__DOUBLE2HIINT
// __DOUBLE2HIINT: CUDA API:
// __DOUBLE2HIINT-NEXT:   __double2hiint(d /*double*/);
// __DOUBLE2HIINT-NEXT: Is migrated to:
// __DOUBLE2HIINT-NEXT:   dpct::cast_double_to_int(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2int_rd | FileCheck %s -check-prefix=__DOUBLE2INT_RD
// __DOUBLE2INT_RD: CUDA API:
// __DOUBLE2INT_RD-NEXT:   __double2int_rd(d /*double*/);
// __DOUBLE2INT_RD-NEXT: Is migrated to:
// __DOUBLE2INT_RD-NEXT:   sycl::vec<double, 1>{d}.convert<int, sycl::rounding_mode::rtn>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2int_rn | FileCheck %s -check-prefix=__DOUBLE2INT_RN
// __DOUBLE2INT_RN: CUDA API:
// __DOUBLE2INT_RN-NEXT:   __double2int_rn(d /*double*/);
// __DOUBLE2INT_RN-NEXT: Is migrated to:
// __DOUBLE2INT_RN-NEXT:   sycl::vec<double, 1>{d}.convert<int, sycl::rounding_mode::rte>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2int_ru | FileCheck %s -check-prefix=__DOUBLE2INT_RU
// __DOUBLE2INT_RU: CUDA API:
// __DOUBLE2INT_RU-NEXT:   __double2int_ru(d /*double*/);
// __DOUBLE2INT_RU-NEXT: Is migrated to:
// __DOUBLE2INT_RU-NEXT:   sycl::vec<double, 1>{d}.convert<int, sycl::rounding_mode::rtp>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2int_rz | FileCheck %s -check-prefix=__DOUBLE2INT_RZ
// __DOUBLE2INT_RZ: CUDA API:
// __DOUBLE2INT_RZ-NEXT:   __double2int_rz(d /*double*/);
// __DOUBLE2INT_RZ-NEXT: Is migrated to:
// __DOUBLE2INT_RZ-NEXT:   sycl::vec<double, 1>{d}.convert<int, sycl::rounding_mode::rtz>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2ll_rd | FileCheck %s -check-prefix=__DOUBLE2LL_RD
// __DOUBLE2LL_RD: CUDA API:
// __DOUBLE2LL_RD-NEXT:   __double2ll_rd(d /*double*/);
// __DOUBLE2LL_RD-NEXT: Is migrated to:
// __DOUBLE2LL_RD-NEXT:   sycl::vec<double, 1>{d}.convert<long long, sycl::rounding_mode::rtn>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2ll_rn | FileCheck %s -check-prefix=__DOUBLE2LL_RN
// __DOUBLE2LL_RN: CUDA API:
// __DOUBLE2LL_RN-NEXT:   __double2ll_rn(d /*double*/);
// __DOUBLE2LL_RN-NEXT: Is migrated to:
// __DOUBLE2LL_RN-NEXT:   sycl::vec<double, 1>{d}.convert<long long, sycl::rounding_mode::rte>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2ll_ru | FileCheck %s -check-prefix=__DOUBLE2LL_RU
// __DOUBLE2LL_RU: CUDA API:
// __DOUBLE2LL_RU-NEXT:   __double2ll_ru(d /*double*/);
// __DOUBLE2LL_RU-NEXT: Is migrated to:
// __DOUBLE2LL_RU-NEXT:   sycl::vec<double, 1>{d}.convert<long long, sycl::rounding_mode::rtp>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2ll_rz | FileCheck %s -check-prefix=__DOUBLE2LL_RZ
// __DOUBLE2LL_RZ: CUDA API:
// __DOUBLE2LL_RZ-NEXT:   __double2ll_rz(d /*double*/);
// __DOUBLE2LL_RZ-NEXT: Is migrated to:
// __DOUBLE2LL_RZ-NEXT:   sycl::vec<double, 1>{d}.convert<long long, sycl::rounding_mode::rtz>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2loint | FileCheck %s -check-prefix=__DOUBLE2LOINT
// __DOUBLE2LOINT: CUDA API:
// __DOUBLE2LOINT-NEXT:   __double2loint(d /*double*/);
// __DOUBLE2LOINT-NEXT: Is migrated to:
// __DOUBLE2LOINT-NEXT:   dpct::cast_double_to_int(d, false);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2uint_rd | FileCheck %s -check-prefix=__DOUBLE2UINT_RD
// __DOUBLE2UINT_RD: CUDA API:
// __DOUBLE2UINT_RD-NEXT:   __double2uint_rd(d /*double*/);
// __DOUBLE2UINT_RD-NEXT: Is migrated to:
// __DOUBLE2UINT_RD-NEXT:   sycl::vec<double, 1>{d}.convert<unsigned int, sycl::rounding_mode::rtn>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2uint_rn | FileCheck %s -check-prefix=__DOUBLE2UINT_RN
// __DOUBLE2UINT_RN: CUDA API:
// __DOUBLE2UINT_RN-NEXT:   __double2uint_rn(d /*double*/);
// __DOUBLE2UINT_RN-NEXT: Is migrated to:
// __DOUBLE2UINT_RN-NEXT:   sycl::vec<double, 1>{d}.convert<unsigned int, sycl::rounding_mode::rte>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2uint_ru | FileCheck %s -check-prefix=__DOUBLE2UINT_RU
// __DOUBLE2UINT_RU: CUDA API:
// __DOUBLE2UINT_RU-NEXT:   __double2uint_ru(d /*double*/);
// __DOUBLE2UINT_RU-NEXT: Is migrated to:
// __DOUBLE2UINT_RU-NEXT:   sycl::vec<double, 1>{d}.convert<unsigned int, sycl::rounding_mode::rtp>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2uint_rz | FileCheck %s -check-prefix=__DOUBLE2UINT_RZ
// __DOUBLE2UINT_RZ: CUDA API:
// __DOUBLE2UINT_RZ-NEXT:   __double2uint_rz(d /*double*/);
// __DOUBLE2UINT_RZ-NEXT: Is migrated to:
// __DOUBLE2UINT_RZ-NEXT:   sycl::vec<double, 1>{d}.convert<unsigned int, sycl::rounding_mode::rtz>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2ull_rd | FileCheck %s -check-prefix=__DOUBLE2ULL_RD
// __DOUBLE2ULL_RD: CUDA API:
// __DOUBLE2ULL_RD-NEXT:   __double2ull_rd(d /*double*/);
// __DOUBLE2ULL_RD-NEXT: Is migrated to:
// __DOUBLE2ULL_RD-NEXT:   sycl::vec<double, 1>{d}.convert<unsigned long long, sycl::rounding_mode::rtn>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2ull_rn | FileCheck %s -check-prefix=__DOUBLE2ULL_RN
// __DOUBLE2ULL_RN: CUDA API:
// __DOUBLE2ULL_RN-NEXT:   __double2ull_rn(d /*double*/);
// __DOUBLE2ULL_RN-NEXT: Is migrated to:
// __DOUBLE2ULL_RN-NEXT:   sycl::vec<double, 1>{d}.convert<unsigned long long, sycl::rounding_mode::rte>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2ull_ru | FileCheck %s -check-prefix=__DOUBLE2ULL_RU
// __DOUBLE2ULL_RU: CUDA API:
// __DOUBLE2ULL_RU-NEXT:   __double2ull_ru(d /*double*/);
// __DOUBLE2ULL_RU-NEXT: Is migrated to:
// __DOUBLE2ULL_RU-NEXT:   sycl::vec<double, 1>{d}.convert<unsigned long long, sycl::rounding_mode::rtp>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2ull_rz | FileCheck %s -check-prefix=__DOUBLE2ULL_RZ
// __DOUBLE2ULL_RZ: CUDA API:
// __DOUBLE2ULL_RZ-NEXT:   __double2ull_rz(d /*double*/);
// __DOUBLE2ULL_RZ-NEXT: Is migrated to:
// __DOUBLE2ULL_RZ-NEXT:   sycl::vec<double, 1>{d}.convert<unsigned long long, sycl::rounding_mode::rtz>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double_as_longlong | FileCheck %s -check-prefix=__DOUBLE_AS_LONGLONG
// __DOUBLE_AS_LONGLONG: CUDA API:
// __DOUBLE_AS_LONGLONG-NEXT:   __double_as_longlong(d /*double*/);
// __DOUBLE_AS_LONGLONG-NEXT: Is migrated to:
// __DOUBLE_AS_LONGLONG-NEXT:   sycl::bit_cast<long long>(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2int_rd | FileCheck %s -check-prefix=__FLOAT2INT_RD
// __FLOAT2INT_RD: CUDA API:
// __FLOAT2INT_RD-NEXT:   __float2int_rd(d /*float*/);
// __FLOAT2INT_RD-NEXT: Is migrated to:
// __FLOAT2INT_RD-NEXT:   sycl::vec<float, 1>{d}.convert<int, sycl::rounding_mode::rtn>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2int_rn | FileCheck %s -check-prefix=__FLOAT2INT_RN
// __FLOAT2INT_RN: CUDA API:
// __FLOAT2INT_RN-NEXT:   __float2int_rn(d /*float*/);
// __FLOAT2INT_RN-NEXT: Is migrated to:
// __FLOAT2INT_RN-NEXT:   sycl::vec<float, 1>{d}.convert<int, sycl::rounding_mode::rte>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2int_ru | FileCheck %s -check-prefix=__FLOAT2INT_RU
// __FLOAT2INT_RU: CUDA API:
// __FLOAT2INT_RU-NEXT:   __float2int_ru(d /*float*/);
// __FLOAT2INT_RU-NEXT: Is migrated to:
// __FLOAT2INT_RU-NEXT:   sycl::vec<float, 1>{d}.convert<int, sycl::rounding_mode::rtp>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2int_rz | FileCheck %s -check-prefix=__FLOAT2INT_RZ
// __FLOAT2INT_RZ: CUDA API:
// __FLOAT2INT_RZ-NEXT:   __float2int_rz(d /*float*/);
// __FLOAT2INT_RZ-NEXT: Is migrated to:
// __FLOAT2INT_RZ-NEXT:   sycl::vec<float, 1>{d}.convert<int, sycl::rounding_mode::rtz>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2ll_rd | FileCheck %s -check-prefix=__FLOAT2LL_RD
// __FLOAT2LL_RD: CUDA API:
// __FLOAT2LL_RD-NEXT:   __float2ll_rd(d /*float*/);
// __FLOAT2LL_RD-NEXT: Is migrated to:
// __FLOAT2LL_RD-NEXT:   sycl::vec<float, 1>{d}.convert<long long, sycl::rounding_mode::rtn>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2ll_rn | FileCheck %s -check-prefix=__FLOAT2LL_RN
// __FLOAT2LL_RN: CUDA API:
// __FLOAT2LL_RN-NEXT:   __float2ll_rn(d /*float*/);
// __FLOAT2LL_RN-NEXT: Is migrated to:
// __FLOAT2LL_RN-NEXT:   sycl::vec<float, 1>{d}.convert<long long, sycl::rounding_mode::rte>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2ll_ru | FileCheck %s -check-prefix=__FLOAT2LL_RU
// __FLOAT2LL_RU: CUDA API:
// __FLOAT2LL_RU-NEXT:   __float2ll_ru(d /*float*/);
// __FLOAT2LL_RU-NEXT: Is migrated to:
// __FLOAT2LL_RU-NEXT:   sycl::vec<float, 1>{d}.convert<long long, sycl::rounding_mode::rtp>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2ll_rz | FileCheck %s -check-prefix=__FLOAT2LL_RZ
// __FLOAT2LL_RZ: CUDA API:
// __FLOAT2LL_RZ-NEXT:   __float2ll_rz(d /*float*/);
// __FLOAT2LL_RZ-NEXT: Is migrated to:
// __FLOAT2LL_RZ-NEXT:   sycl::vec<float, 1>{d}.convert<long long, sycl::rounding_mode::rtz>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2uint_rd | FileCheck %s -check-prefix=__FLOAT2UINT_RD
// __FLOAT2UINT_RD: CUDA API:
// __FLOAT2UINT_RD-NEXT:   __float2uint_rd(d /*float*/);
// __FLOAT2UINT_RD-NEXT: Is migrated to:
// __FLOAT2UINT_RD-NEXT:   sycl::vec<float, 1>{d}.convert<unsigned int, sycl::rounding_mode::rtn>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2uint_rn | FileCheck %s -check-prefix=__FLOAT2UINT_RN
// __FLOAT2UINT_RN: CUDA API:
// __FLOAT2UINT_RN-NEXT:   __float2uint_rn(d /*float*/);
// __FLOAT2UINT_RN-NEXT: Is migrated to:
// __FLOAT2UINT_RN-NEXT:   sycl::vec<float, 1>{d}.convert<unsigned int, sycl::rounding_mode::rte>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2uint_ru | FileCheck %s -check-prefix=__FLOAT2UINT_RU
// __FLOAT2UINT_RU: CUDA API:
// __FLOAT2UINT_RU-NEXT:   __float2uint_ru(d /*float*/);
// __FLOAT2UINT_RU-NEXT: Is migrated to:
// __FLOAT2UINT_RU-NEXT:   sycl::vec<float, 1>{d}.convert<unsigned int, sycl::rounding_mode::rtp>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2uint_rz | FileCheck %s -check-prefix=__FLOAT2UINT_RZ
// __FLOAT2UINT_RZ: CUDA API:
// __FLOAT2UINT_RZ-NEXT:   __float2uint_rz(d /*float*/);
// __FLOAT2UINT_RZ-NEXT: Is migrated to:
// __FLOAT2UINT_RZ-NEXT:   sycl::vec<float, 1>{d}.convert<unsigned int, sycl::rounding_mode::rtz>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2ull_rd | FileCheck %s -check-prefix=__FLOAT2ULL_RD
// __FLOAT2ULL_RD: CUDA API:
// __FLOAT2ULL_RD-NEXT:   __float2ull_rd(d /*float*/);
// __FLOAT2ULL_RD-NEXT: Is migrated to:
// __FLOAT2ULL_RD-NEXT:   sycl::vec<float, 1>{d}.convert<unsigned long long, sycl::rounding_mode::rtn>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2ull_rn | FileCheck %s -check-prefix=__FLOAT2ULL_RN
// __FLOAT2ULL_RN: CUDA API:
// __FLOAT2ULL_RN-NEXT:   __float2ull_rn(d /*float*/);
// __FLOAT2ULL_RN-NEXT: Is migrated to:
// __FLOAT2ULL_RN-NEXT:   sycl::vec<float, 1>{d}.convert<unsigned long long, sycl::rounding_mode::rte>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2ull_ru | FileCheck %s -check-prefix=__FLOAT2ULL_RU
// __FLOAT2ULL_RU: CUDA API:
// __FLOAT2ULL_RU-NEXT:   __float2ull_ru(d /*float*/);
// __FLOAT2ULL_RU-NEXT: Is migrated to:
// __FLOAT2ULL_RU-NEXT:   sycl::vec<float, 1>{d}.convert<unsigned long long, sycl::rounding_mode::rtp>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2ull_rz | FileCheck %s -check-prefix=__FLOAT2ULL_RZ
// __FLOAT2ULL_RZ: CUDA API:
// __FLOAT2ULL_RZ-NEXT:   __float2ull_rz(d /*float*/);
// __FLOAT2ULL_RZ-NEXT: Is migrated to:
// __FLOAT2ULL_RZ-NEXT:   sycl::vec<float, 1>{d}.convert<unsigned long long, sycl::rounding_mode::rtz>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float_as_int | FileCheck %s -check-prefix=__FLOAT_AS_INT
// __FLOAT_AS_INT: CUDA API:
// __FLOAT_AS_INT-NEXT:   __float_as_int(d /*float*/);
// __FLOAT_AS_INT-NEXT: Is migrated to:
// __FLOAT_AS_INT-NEXT:   sycl::bit_cast<int>(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float_as_uint | FileCheck %s -check-prefix=__FLOAT_AS_UINT
// __FLOAT_AS_UINT: CUDA API:
// __FLOAT_AS_UINT-NEXT:   __float_as_uint(d /*float*/);
// __FLOAT_AS_UINT-NEXT: Is migrated to:
// __FLOAT_AS_UINT-NEXT:   sycl::bit_cast<unsigned int>(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hiloint2double | FileCheck %s -check-prefix=__HILOINT2DOUBLE
// __HILOINT2DOUBLE: CUDA API:
// __HILOINT2DOUBLE-NEXT:   __hiloint2double(i1 /*int*/, i2 /*int*/);
// __HILOINT2DOUBLE-NEXT: Is migrated to:
// __HILOINT2DOUBLE-NEXT:   dpct::cast_ints_to_double(i1, i2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__int2double_rn | FileCheck %s -check-prefix=__INT2DOUBLE_RN
// __INT2DOUBLE_RN: CUDA API:
// __INT2DOUBLE_RN-NEXT:   __int2double_rn(i /*int*/);
// __INT2DOUBLE_RN-NEXT: Is migrated to:
// __INT2DOUBLE_RN-NEXT:   sycl::vec<int, 1>{i}.convert<double, sycl::rounding_mode::rte>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__int2float_rd | FileCheck %s -check-prefix=__INT2FLOAT_RD
// __INT2FLOAT_RD: CUDA API:
// __INT2FLOAT_RD-NEXT:   __int2float_rd(i /*int*/);
// __INT2FLOAT_RD-NEXT: Is migrated to:
// __INT2FLOAT_RD-NEXT:   sycl::vec<int, 1>{i}.convert<float, sycl::rounding_mode::rtn>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__int2float_rn | FileCheck %s -check-prefix=__INT2FLOAT_RN
// __INT2FLOAT_RN: CUDA API:
// __INT2FLOAT_RN-NEXT:   __int2float_rn(i /*int*/);
// __INT2FLOAT_RN-NEXT: Is migrated to:
// __INT2FLOAT_RN-NEXT:   sycl::vec<int, 1>{i}.convert<float, sycl::rounding_mode::rte>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__int2float_ru | FileCheck %s -check-prefix=__INT2FLOAT_RU
// __INT2FLOAT_RU: CUDA API:
// __INT2FLOAT_RU-NEXT:   __int2float_ru(i /*int*/);
// __INT2FLOAT_RU-NEXT: Is migrated to:
// __INT2FLOAT_RU-NEXT:   sycl::vec<int, 1>{i}.convert<float, sycl::rounding_mode::rtp>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__int2float_rz | FileCheck %s -check-prefix=__INT2FLOAT_RZ
// __INT2FLOAT_RZ: CUDA API:
// __INT2FLOAT_RZ-NEXT:   __int2float_rz(i /*int*/);
// __INT2FLOAT_RZ-NEXT: Is migrated to:
// __INT2FLOAT_RZ-NEXT:   sycl::vec<int, 1>{i}.convert<float, sycl::rounding_mode::rtz>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__int_as_float | FileCheck %s -check-prefix=__INT_AS_FLOAT
// __INT_AS_FLOAT: CUDA API:
// __INT_AS_FLOAT-NEXT:   __int_as_float(i /*int*/);
// __INT_AS_FLOAT-NEXT: Is migrated to:
// __INT_AS_FLOAT-NEXT:   sycl::bit_cast<float>(i);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ll2double_rd | FileCheck %s -check-prefix=__LL2DOUBLE_RD
// __LL2DOUBLE_RD: CUDA API:
// __LL2DOUBLE_RD-NEXT:   __ll2double_rd(ll /*long long int*/);
// __LL2DOUBLE_RD-NEXT: Is migrated to:
// __LL2DOUBLE_RD-NEXT:   sycl::vec<long long, 1>{ll}.convert<double, sycl::rounding_mode::rtn>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ll2double_rn | FileCheck %s -check-prefix=__LL2DOUBLE_RN
// __LL2DOUBLE_RN: CUDA API:
// __LL2DOUBLE_RN-NEXT:   __ll2double_rn(ll /*long long int*/);
// __LL2DOUBLE_RN-NEXT: Is migrated to:
// __LL2DOUBLE_RN-NEXT:   sycl::vec<long long, 1>{ll}.convert<double, sycl::rounding_mode::rte>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ll2double_ru | FileCheck %s -check-prefix=__LL2DOUBLE_RU
// __LL2DOUBLE_RU: CUDA API:
// __LL2DOUBLE_RU-NEXT:   __ll2double_ru(ll /*long long int*/);
// __LL2DOUBLE_RU-NEXT: Is migrated to:
// __LL2DOUBLE_RU-NEXT:   sycl::vec<long long, 1>{ll}.convert<double, sycl::rounding_mode::rtp>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ll2double_rz | FileCheck %s -check-prefix=__LL2DOUBLE_RZ
// __LL2DOUBLE_RZ: CUDA API:
// __LL2DOUBLE_RZ-NEXT:   __ll2double_rz(ll /*long long int*/);
// __LL2DOUBLE_RZ-NEXT: Is migrated to:
// __LL2DOUBLE_RZ-NEXT:   sycl::vec<long long, 1>{ll}.convert<double, sycl::rounding_mode::rtz>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ll2float_rd | FileCheck %s -check-prefix=__LL2FLOAT_RD
// __LL2FLOAT_RD: CUDA API:
// __LL2FLOAT_RD-NEXT:   __ll2float_rd(ll /*long long int*/);
// __LL2FLOAT_RD-NEXT: Is migrated to:
// __LL2FLOAT_RD-NEXT:   sycl::vec<long long, 1>{ll}.convert<float, sycl::rounding_mode::rtn>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ll2float_rn | FileCheck %s -check-prefix=__LL2FLOAT_RN
// __LL2FLOAT_RN: CUDA API:
// __LL2FLOAT_RN-NEXT:   __ll2float_rn(ll /*long long int*/);
// __LL2FLOAT_RN-NEXT: Is migrated to:
// __LL2FLOAT_RN-NEXT:   sycl::vec<long long, 1>{ll}.convert<float, sycl::rounding_mode::rte>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ll2float_ru | FileCheck %s -check-prefix=__LL2FLOAT_RU
// __LL2FLOAT_RU: CUDA API:
// __LL2FLOAT_RU-NEXT:   __ll2float_ru(ll /*long long int*/);
// __LL2FLOAT_RU-NEXT: Is migrated to:
// __LL2FLOAT_RU-NEXT:   sycl::vec<long long, 1>{ll}.convert<float, sycl::rounding_mode::rtp>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ll2float_rz | FileCheck %s -check-prefix=__LL2FLOAT_RZ
// __LL2FLOAT_RZ: CUDA API:
// __LL2FLOAT_RZ-NEXT:   __ll2float_rz(ll /*long long int*/);
// __LL2FLOAT_RZ-NEXT: Is migrated to:
// __LL2FLOAT_RZ-NEXT:   sycl::vec<long long, 1>{ll}.convert<float, sycl::rounding_mode::rtz>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__longlong_as_double | FileCheck %s -check-prefix=__LONGLONG_AS_DOUBLE
// __LONGLONG_AS_DOUBLE: CUDA API:
// __LONGLONG_AS_DOUBLE-NEXT:   __longlong_as_double(ll /*long long int*/);
// __LONGLONG_AS_DOUBLE-NEXT: Is migrated to:
// __LONGLONG_AS_DOUBLE-NEXT:   sycl::bit_cast<double>(ll);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__uint2double_rn | FileCheck %s -check-prefix=__UINT2DOUBLE_RN
// __UINT2DOUBLE_RN: CUDA API:
// __UINT2DOUBLE_RN-NEXT:   __uint2double_rn(u /*unsigned int*/);
// __UINT2DOUBLE_RN-NEXT: Is migrated to:
// __UINT2DOUBLE_RN-NEXT:   sycl::vec<unsigned int, 1>{u}.convert<double, sycl::rounding_mode::rte>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__uint2float_rd | FileCheck %s -check-prefix=__UINT2FLOAT_RD
// __UINT2FLOAT_RD: CUDA API:
// __UINT2FLOAT_RD-NEXT:   __uint2float_rd(u /*unsigned int*/);
// __UINT2FLOAT_RD-NEXT: Is migrated to:
// __UINT2FLOAT_RD-NEXT:   sycl::vec<unsigned int, 1>{u}.convert<float, sycl::rounding_mode::rtn>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__uint2float_rn | FileCheck %s -check-prefix=__UINT2FLOAT_RN
// __UINT2FLOAT_RN: CUDA API:
// __UINT2FLOAT_RN-NEXT:   __uint2float_rn(u /*unsigned int*/);
// __UINT2FLOAT_RN-NEXT: Is migrated to:
// __UINT2FLOAT_RN-NEXT:   sycl::vec<unsigned int, 1>{u}.convert<float, sycl::rounding_mode::rte>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__uint2float_ru | FileCheck %s -check-prefix=__UINT2FLOAT_RU
// __UINT2FLOAT_RU: CUDA API:
// __UINT2FLOAT_RU-NEXT:   __uint2float_ru(u /*unsigned int*/);
// __UINT2FLOAT_RU-NEXT: Is migrated to:
// __UINT2FLOAT_RU-NEXT:   sycl::vec<unsigned int, 1>{u}.convert<float, sycl::rounding_mode::rtp>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__uint2float_rz | FileCheck %s -check-prefix=__UINT2FLOAT_RZ
// __UINT2FLOAT_RZ: CUDA API:
// __UINT2FLOAT_RZ-NEXT:   __uint2float_rz(u /*unsigned int*/);
// __UINT2FLOAT_RZ-NEXT: Is migrated to:
// __UINT2FLOAT_RZ-NEXT:   sycl::vec<unsigned int, 1>{u}.convert<float, sycl::rounding_mode::rtz>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__uint_as_float | FileCheck %s -check-prefix=__UINT_AS_FLOAT
// __UINT_AS_FLOAT: CUDA API:
// __UINT_AS_FLOAT-NEXT:   __uint_as_float(u /*unsigned int*/);
// __UINT_AS_FLOAT-NEXT: Is migrated to:
// __UINT_AS_FLOAT-NEXT:   sycl::bit_cast<float>(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ull2double_rd | FileCheck %s -check-prefix=__ULL2DOUBLE_RD
// __ULL2DOUBLE_RD: CUDA API:
// __ULL2DOUBLE_RD-NEXT:   __ull2double_rd(ull /*unsigned long long int*/);
// __ULL2DOUBLE_RD-NEXT: Is migrated to:
// __ULL2DOUBLE_RD-NEXT:   sycl::vec<unsigned long long, 1>{ull}.convert<double, sycl::rounding_mode::rtn>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ull2double_rn | FileCheck %s -check-prefix=__ULL2DOUBLE_RN
// __ULL2DOUBLE_RN: CUDA API:
// __ULL2DOUBLE_RN-NEXT:   __ull2double_rn(ull /*unsigned long long int*/);
// __ULL2DOUBLE_RN-NEXT: Is migrated to:
// __ULL2DOUBLE_RN-NEXT:   sycl::vec<unsigned long long, 1>{ull}.convert<double, sycl::rounding_mode::rte>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ull2double_ru | FileCheck %s -check-prefix=__ULL2DOUBLE_RU
// __ULL2DOUBLE_RU: CUDA API:
// __ULL2DOUBLE_RU-NEXT:   __ull2double_ru(ull /*unsigned long long int*/);
// __ULL2DOUBLE_RU-NEXT: Is migrated to:
// __ULL2DOUBLE_RU-NEXT:   sycl::vec<unsigned long long, 1>{ull}.convert<double, sycl::rounding_mode::rtp>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ull2double_rz | FileCheck %s -check-prefix=__ULL2DOUBLE_RZ
// __ULL2DOUBLE_RZ: CUDA API:
// __ULL2DOUBLE_RZ-NEXT:   __ull2double_rz(ull /*unsigned long long int*/);
// __ULL2DOUBLE_RZ-NEXT: Is migrated to:
// __ULL2DOUBLE_RZ-NEXT:   sycl::vec<unsigned long long, 1>{ull}.convert<double, sycl::rounding_mode::rtz>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ull2float_rd | FileCheck %s -check-prefix=__ULL2FLOAT_RD
// __ULL2FLOAT_RD: CUDA API:
// __ULL2FLOAT_RD-NEXT:   __ull2float_rd(ull /*unsigned long long int*/);
// __ULL2FLOAT_RD-NEXT: Is migrated to:
// __ULL2FLOAT_RD-NEXT:   sycl::vec<unsigned long long, 1>{ull}.convert<float, sycl::rounding_mode::rtn>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ull2float_rn | FileCheck %s -check-prefix=__ULL2FLOAT_RN
// __ULL2FLOAT_RN: CUDA API:
// __ULL2FLOAT_RN-NEXT:   __ull2float_rn(ull /*unsigned long long int*/);
// __ULL2FLOAT_RN-NEXT: Is migrated to:
// __ULL2FLOAT_RN-NEXT:   sycl::vec<unsigned long long, 1>{ull}.convert<float, sycl::rounding_mode::rte>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ull2float_ru | FileCheck %s -check-prefix=__ULL2FLOAT_RU
// __ULL2FLOAT_RU: CUDA API:
// __ULL2FLOAT_RU-NEXT:   __ull2float_ru(ull /*unsigned long long int*/);
// __ULL2FLOAT_RU-NEXT: Is migrated to:
// __ULL2FLOAT_RU-NEXT:   sycl::vec<unsigned long long, 1>{ull}.convert<float, sycl::rounding_mode::rtp>()[0];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ull2float_rz | FileCheck %s -check-prefix=__ULL2FLOAT_RZ
// __ULL2FLOAT_RZ: CUDA API:
// __ULL2FLOAT_RZ-NEXT:   __ull2float_rz(ull /*unsigned long long int*/);
// __ULL2FLOAT_RZ-NEXT: Is migrated to:
// __ULL2FLOAT_RZ-NEXT:   sycl::vec<unsigned long long, 1>{ull}.convert<float, sycl::rounding_mode::rtz>()[0];
