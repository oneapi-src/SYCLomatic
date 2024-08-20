// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2

/// Built-in Vector Types

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_char1 | FileCheck %s -check-prefix=MAKE_CHAR1
// MAKE_CHAR1: CUDA API:
// MAKE_CHAR1-NEXT:   make_char1(c /*char*/);
// MAKE_CHAR1-NEXT: Is migrated to:
// MAKE_CHAR1-NEXT:   int8_t(c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_char2 | FileCheck %s -check-prefix=MAKE_CHAR2
// MAKE_CHAR2: CUDA API:
// MAKE_CHAR2-NEXT:   make_char2(c1 /*char*/, c2 /*char*/);
// MAKE_CHAR2-NEXT: Is migrated to:
// MAKE_CHAR2-NEXT:   sycl::char2(c1, c2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_char3 | FileCheck %s -check-prefix=MAKE_CHAR3
// MAKE_CHAR3: CUDA API:
// MAKE_CHAR3-NEXT:   make_char3(c1 /*char*/, c2 /*char*/, c3 /*char*/);
// MAKE_CHAR3-NEXT: Is migrated to:
// MAKE_CHAR3-NEXT:   sycl::char3(c1, c2, c3);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_char4 | FileCheck %s -check-prefix=MAKE_CHAR4
// MAKE_CHAR4: CUDA API:
// MAKE_CHAR4-NEXT:   make_char4(c1 /*char*/, c2 /*char*/, c3 /*char*/, c4 /*char*/);
// MAKE_CHAR4-NEXT: Is migrated to:
// MAKE_CHAR4-NEXT:   sycl::char4(c1, c2, c3, c4);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_uchar1 | FileCheck %s -check-prefix=MAKE_UCHAR1
// MAKE_UCHAR1: CUDA API:
// MAKE_UCHAR1-NEXT:   make_uchar1(u /*unsigned char*/);
// MAKE_UCHAR1-NEXT: Is migrated to:
// MAKE_UCHAR1-NEXT:   uint8_t(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_uchar2 | FileCheck %s -check-prefix=MAKE_UCHAR2
// MAKE_UCHAR2: CUDA API:
// MAKE_UCHAR2-NEXT:   make_uchar2(u1 /*unsigned char*/, u2 /*unsigned char*/);
// MAKE_UCHAR2-NEXT: Is migrated to:
// MAKE_UCHAR2-NEXT:   sycl::uchar2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_uchar3 | FileCheck %s -check-prefix=MAKE_UCHAR3
// MAKE_UCHAR3: CUDA API:
// MAKE_UCHAR3-NEXT:   make_uchar3(u1 /*unsigned char*/, u2 /*unsigned char*/, u3 /*unsigned char*/);
// MAKE_UCHAR3-NEXT: Is migrated to:
// MAKE_UCHAR3-NEXT:   sycl::uchar3(u1, u2, u3);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_uchar4 | FileCheck %s -check-prefix=MAKE_UCHAR4
// MAKE_UCHAR4: CUDA API:
// MAKE_UCHAR4-NEXT:   make_uchar4(u1 /*unsigned char*/, u2 /*unsigned char*/, u3 /*unsigned char*/,
// MAKE_UCHAR4-NEXT:               u4 /*unsigned char*/);
// MAKE_UCHAR4-NEXT: Is migrated to:
// MAKE_UCHAR4-NEXT:   sycl::uchar4(u1, u2, u3, u4);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_short1 | FileCheck %s -check-prefix=MAKE_SHORT1
// MAKE_SHORT1: CUDA API:
// MAKE_SHORT1-NEXT:   make_short1(s /*short*/);
// MAKE_SHORT1-NEXT: Is migrated to:
// MAKE_SHORT1-NEXT:   int16_t(s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_short2 | FileCheck %s -check-prefix=MAKE_SHORT2
// MAKE_SHORT2: CUDA API:
// MAKE_SHORT2-NEXT:   make_short2(s1 /*short*/, s2 /*short*/);
// MAKE_SHORT2-NEXT: Is migrated to:
// MAKE_SHORT2-NEXT:   sycl::short2(s1, s2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_short3 | FileCheck %s -check-prefix=MAKE_SHORT3
// MAKE_SHORT3: CUDA API:
// MAKE_SHORT3-NEXT:   make_short3(s1 /*short*/, s2 /*short*/, s3 /*short*/);
// MAKE_SHORT3-NEXT: Is migrated to:
// MAKE_SHORT3-NEXT:   sycl::short3(s1, s2, s3);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_short4 | FileCheck %s -check-prefix=MAKE_SHORT4
// MAKE_SHORT4: CUDA API:
// MAKE_SHORT4-NEXT:   make_short4(s1 /*short*/, s2 /*short*/, s3 /*short*/, s4 /*short*/);
// MAKE_SHORT4-NEXT: Is migrated to:
// MAKE_SHORT4-NEXT:   sycl::short4(s1, s2, s3, s4);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_ushort1 | FileCheck %s -check-prefix=MAKE_USHORT1
// MAKE_USHORT1: CUDA API:
// MAKE_USHORT1-NEXT:   make_ushort1(u /*unsigned short*/);
// MAKE_USHORT1-NEXT: Is migrated to:
// MAKE_USHORT1-NEXT:   uint16_t(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_ushort2 | FileCheck %s -check-prefix=MAKE_USHORT2
// MAKE_USHORT2: CUDA API:
// MAKE_USHORT2-NEXT:   make_ushort2(u1 /*unsigned short*/, u2 /*unsigned short*/);
// MAKE_USHORT2-NEXT: Is migrated to:
// MAKE_USHORT2-NEXT:   sycl::ushort2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_ushort3 | FileCheck %s -check-prefix=MAKE_USHORT3
// MAKE_USHORT3: CUDA API:
// MAKE_USHORT3-NEXT:   make_ushort3(u1 /*unsigned short*/, u2 /*unsigned short*/,
// MAKE_USHORT3-NEXT:                u3 /*unsigned short*/);
// MAKE_USHORT3-NEXT: Is migrated to:
// MAKE_USHORT3-NEXT:   sycl::ushort3(u1, u2, u3);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_ushort4 | FileCheck %s -check-prefix=MAKE_USHORT4
// MAKE_USHORT4: CUDA API:
// MAKE_USHORT4-NEXT:   make_ushort4(u1 /*unsigned short*/, u2 /*unsigned short*/,
// MAKE_USHORT4-NEXT:                u3 /*unsigned short*/, u4 /*unsigned short*/);
// MAKE_USHORT4-NEXT: Is migrated to:
// MAKE_USHORT4-NEXT:   sycl::ushort4(u1, u2, u3, u4);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_int1 | FileCheck %s -check-prefix=MAKE_INT1
// MAKE_INT1: CUDA API:
// MAKE_INT1-NEXT:   make_int1(i /*int*/);
// MAKE_INT1-NEXT: Is migrated to:
// MAKE_INT1-NEXT:   int32_t(i);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_int2 | FileCheck %s -check-prefix=MAKE_INT2
// MAKE_INT2: CUDA API:
// MAKE_INT2-NEXT:   make_int2(i1 /*int*/, i2 /*int*/);
// MAKE_INT2-NEXT: Is migrated to:
// MAKE_INT2-NEXT:   sycl::int2(i1, i2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_int3 | FileCheck %s -check-prefix=MAKE_INT3
// MAKE_INT3: CUDA API:
// MAKE_INT3-NEXT:   make_int3(i1 /*int*/, i2 /*int*/, i3 /*int*/);
// MAKE_INT3-NEXT: Is migrated to:
// MAKE_INT3-NEXT:   sycl::int3(i1, i2, i3);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_int4 | FileCheck %s -check-prefix=MAKE_INT4
// MAKE_INT4: CUDA API:
// MAKE_INT4-NEXT:   make_int4(i1 /*int*/, i2 /*int*/, i3 /*int*/, i4 /*int*/);
// MAKE_INT4-NEXT: Is migrated to:
// MAKE_INT4-NEXT:   sycl::int4(i1, i2, i3, i4);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_uint1 | FileCheck %s -check-prefix=MAKE_UINT1
// MAKE_UINT1: CUDA API:
// MAKE_UINT1-NEXT:   make_uint1(u /*unsigned int*/);
// MAKE_UINT1-NEXT: Is migrated to:
// MAKE_UINT1-NEXT:   uint32_t(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_uint2 | FileCheck %s -check-prefix=MAKE_UINT2
// MAKE_UINT2: CUDA API:
// MAKE_UINT2-NEXT:   make_uint2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// MAKE_UINT2-NEXT: Is migrated to:
// MAKE_UINT2-NEXT:   sycl::uint2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_uint3 | FileCheck %s -check-prefix=MAKE_UINT3
// MAKE_UINT3: CUDA API:
// MAKE_UINT3-NEXT:   make_uint3(u1 /*unsigned int*/, u2 /*unsigned int*/, u3 /*unsigned int*/);
// MAKE_UINT3-NEXT: Is migrated to:
// MAKE_UINT3-NEXT:   sycl::uint3(u1, u2, u3);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_uint4 | FileCheck %s -check-prefix=MAKE_UINT4
// MAKE_UINT4: CUDA API:
// MAKE_UINT4-NEXT:   make_uint4(u1 /*unsigned int*/, u2 /*unsigned int*/, u3 /*unsigned int*/,
// MAKE_UINT4-NEXT:              u4 /*unsigned int*/);
// MAKE_UINT4-NEXT: Is migrated to:
// MAKE_UINT4-NEXT:   sycl::uint4(u1, u2, u3, u4);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_long1 | FileCheck %s -check-prefix=MAKE_LONG1
// MAKE_LONG1: CUDA API:
// MAKE_LONG1-NEXT:   make_long1(l /*long*/);
// MAKE_LONG1-NEXT: Is migrated to:
// MAKE_LONG1-NEXT:   int64_t(l);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_long2 | FileCheck %s -check-prefix=MAKE_LONG2
// MAKE_LONG2: CUDA API:
// MAKE_LONG2-NEXT:   make_long2(l1 /*long*/, l2 /*long*/);
// MAKE_LONG2-NEXT: Is migrated to:
// MAKE_LONG2-NEXT:   sycl::long2(l1, l2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_long3 | FileCheck %s -check-prefix=MAKE_LONG3
// MAKE_LONG3: CUDA API:
// MAKE_LONG3-NEXT:   make_long3(l1 /*long*/, l2 /*long*/, l3 /*long*/);
// MAKE_LONG3-NEXT: Is migrated to:
// MAKE_LONG3-NEXT:   sycl::long3(l1, l2, l3);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_long4 | FileCheck %s -check-prefix=MAKE_LONG4
// MAKE_LONG4: CUDA API:
// MAKE_LONG4-NEXT:   make_long4(l1 /*long*/, l2 /*long*/, l3 /*long*/, l4 /*long*/);
// MAKE_LONG4-NEXT: Is migrated to:
// MAKE_LONG4-NEXT:   sycl::long4(l1, l2, l3, l4);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_ulong1 | FileCheck %s -check-prefix=MAKE_ULONG1
// MAKE_ULONG1: CUDA API:
// MAKE_ULONG1-NEXT:   make_ulong1(u /*unsigned long*/);
// MAKE_ULONG1-NEXT: Is migrated to:
// MAKE_ULONG1-NEXT:   uint64_t(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_ulong2 | FileCheck %s -check-prefix=MAKE_ULONG2
// MAKE_ULONG2: CUDA API:
// MAKE_ULONG2-NEXT:   make_ulong2(u1 /*unsigned long*/, u2 /*unsigned long*/);
// MAKE_ULONG2-NEXT: Is migrated to:
// MAKE_ULONG2-NEXT:   sycl::ulong2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_ulong3 | FileCheck %s -check-prefix=MAKE_ULONG3
// MAKE_ULONG3: CUDA API:
// MAKE_ULONG3-NEXT:   make_ulong3(u1 /*unsigned long*/, u2 /*unsigned long*/, u3 /*unsigned long*/);
// MAKE_ULONG3-NEXT: Is migrated to:
// MAKE_ULONG3-NEXT:   sycl::ulong3(u1, u2, u3);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_ulong4 | FileCheck %s -check-prefix=MAKE_ULONG4
// MAKE_ULONG4: CUDA API:
// MAKE_ULONG4-NEXT:   make_ulong4(u1 /*unsigned long*/, u2 /*unsigned long*/, u3 /*unsigned long*/,
// MAKE_ULONG4-NEXT:               u4 /*unsigned long*/);
// MAKE_ULONG4-NEXT: Is migrated to:
// MAKE_ULONG4-NEXT:   sycl::ulong4(u1, u2, u3, u4);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_longlong1 | FileCheck %s -check-prefix=MAKE_LONGLONG1
// MAKE_LONGLONG1: CUDA API:
// MAKE_LONGLONG1-NEXT:   make_longlong1(l /*long long*/);
// MAKE_LONGLONG1-NEXT: Is migrated to:
// MAKE_LONGLONG1-NEXT:   int64_t(l);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_longlong2 | FileCheck %s -check-prefix=MAKE_LONGLONG2
// MAKE_LONGLONG2: CUDA API:
// MAKE_LONGLONG2-NEXT:   make_longlong2(l1 /*long long*/, l2 /*long long*/);
// MAKE_LONGLONG2-NEXT: Is migrated to:
// MAKE_LONGLONG2-NEXT:   sycl::long2(l1, l2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_longlong3 | FileCheck %s -check-prefix=MAKE_LONGLONG3
// MAKE_LONGLONG3: CUDA API:
// MAKE_LONGLONG3-NEXT:   make_longlong3(l1 /*long long*/, l2 /*long long*/, l3 /*long long*/);
// MAKE_LONGLONG3-NEXT: Is migrated to:
// MAKE_LONGLONG3-NEXT:   sycl::long3(l1, l2, l3);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_longlong4 | FileCheck %s -check-prefix=MAKE_LONGLONG4
// MAKE_LONGLONG4: CUDA API:
// MAKE_LONGLONG4-NEXT:   make_longlong4(l1 /*long long*/, l2 /*long long*/, l3 /*long long*/,
// MAKE_LONGLONG4-NEXT:                  l4 /*long long*/);
// MAKE_LONGLONG4-NEXT: Is migrated to:
// MAKE_LONGLONG4-NEXT:   sycl::long4(l1, l2, l3, l4);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_ulonglong1 | FileCheck %s -check-prefix=MAKE_ULONGLONG1
// MAKE_ULONGLONG1: CUDA API:
// MAKE_ULONGLONG1-NEXT:   make_ulonglong1(u /*unsigned long long*/);
// MAKE_ULONGLONG1-NEXT: Is migrated to:
// MAKE_ULONGLONG1-NEXT:   uint64_t(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_ulonglong2 | FileCheck %s -check-prefix=MAKE_ULONGLONG2
// MAKE_ULONGLONG2: CUDA API:
// MAKE_ULONGLONG2-NEXT:   make_ulonglong2(u1 /*unsigned long long*/, u2 /*unsigned long long*/);
// MAKE_ULONGLONG2-NEXT: Is migrated to:
// MAKE_ULONGLONG2-NEXT:   sycl::ulong2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_ulonglong3 | FileCheck %s -check-prefix=MAKE_ULONGLONG3
// MAKE_ULONGLONG3: CUDA API:
// MAKE_ULONGLONG3-NEXT:   make_ulonglong3(u1 /*unsigned long long*/, u2 /*unsigned long long*/,
// MAKE_ULONGLONG3-NEXT:                   u3 /*unsigned long long*/);
// MAKE_ULONGLONG3-NEXT: Is migrated to:
// MAKE_ULONGLONG3-NEXT:   sycl::ulong3(u1, u2, u3);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_ulonglong4 | FileCheck %s -check-prefix=MAKE_ULONGLONG4
// MAKE_ULONGLONG4: CUDA API:
// MAKE_ULONGLONG4-NEXT:   make_ulonglong4(u1 /*unsigned long long*/, u2 /*unsigned long long*/,
// MAKE_ULONGLONG4-NEXT:                   u3 /*unsigned long long*/, u4 /*unsigned long long*/);
// MAKE_ULONGLONG4-NEXT: Is migrated to:
// MAKE_ULONGLONG4-NEXT:   sycl::ulong4(u1, u2, u3, u4);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_float1 | FileCheck %s -check-prefix=MAKE_FLOAT1
// MAKE_FLOAT1: CUDA API:
// MAKE_FLOAT1-NEXT:   make_float1(f /*float*/);
// MAKE_FLOAT1-NEXT: Is migrated to:
// MAKE_FLOAT1-NEXT:   float(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_float2 | FileCheck %s -check-prefix=MAKE_FLOAT2
// MAKE_FLOAT2: CUDA API:
// MAKE_FLOAT2-NEXT:   make_float2(f1 /*float*/, f2 /*float*/);
// MAKE_FLOAT2-NEXT: Is migrated to:
// MAKE_FLOAT2-NEXT:   sycl::float2(f1, f2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_float3 | FileCheck %s -check-prefix=MAKE_FLOAT3
// MAKE_FLOAT3: CUDA API:
// MAKE_FLOAT3-NEXT:   make_float3(f1 /*float*/, f2 /*float*/, f3 /*float*/);
// MAKE_FLOAT3-NEXT: Is migrated to:
// MAKE_FLOAT3-NEXT:   sycl::float3(f1, f2, f3);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_float4 | FileCheck %s -check-prefix=MAKE_FLOAT4
// MAKE_FLOAT4: CUDA API:
// MAKE_FLOAT4-NEXT:   make_float4(f1 /*float*/, f2 /*float*/, f3 /*float*/, f4 /*float*/);
// MAKE_FLOAT4-NEXT: Is migrated to:
// MAKE_FLOAT4-NEXT:   sycl::float4(f1, f2, f3, f4);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_double1 | FileCheck %s -check-prefix=MAKE_DOUBLE1
// MAKE_DOUBLE1: CUDA API:
// MAKE_DOUBLE1-NEXT:   make_double1(d /*double*/);
// MAKE_DOUBLE1-NEXT: Is migrated to:
// MAKE_DOUBLE1-NEXT:   double(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_double2 | FileCheck %s -check-prefix=MAKE_DOUBLE2
// MAKE_DOUBLE2: CUDA API:
// MAKE_DOUBLE2-NEXT:   make_double2(d1 /*double*/, d2 /*double*/);
// MAKE_DOUBLE2-NEXT: Is migrated to:
// MAKE_DOUBLE2-NEXT:   sycl::double2(d1, d2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_double3 | FileCheck %s -check-prefix=MAKE_DOUBLE3
// MAKE_DOUBLE3: CUDA API:
// MAKE_DOUBLE3-NEXT:   make_double3(d1 /*double*/, d2 /*double*/, d3 /*double*/);
// MAKE_DOUBLE3-NEXT: Is migrated to:
// MAKE_DOUBLE3-NEXT:   sycl::double3(d1, d2, d3);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_double4 | FileCheck %s -check-prefix=MAKE_DOUBLE4
// MAKE_DOUBLE4: CUDA API:
// MAKE_DOUBLE4-NEXT:   make_double4(d1 /*double*/, d2 /*double*/, d3 /*double*/, d4 /*double*/);
// MAKE_DOUBLE4-NEXT: Is migrated to:
// MAKE_DOUBLE4-NEXT:   sycl::double4(d1, d2, d3, d4);
