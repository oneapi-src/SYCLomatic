// RUN: dpct --format-range=none -out-root %T/datatypes_test %s --cuda-include-path="%cuda-path/include" --extra-arg="-std=c++14" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/datatypes_test/datatypes_test.dp.cpp

#include <iostream>
#include <iostream>
#include <cuda.h>
#include <cublas.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cusolverDn.h>
#include <cufft.h>

void case_1(void) {

{
// CHECK: char var1;
// CHECK-NEXT: char *var2;
// CHECK-NEXT: char &var3 = var1;
// CHECK-NEXT: char &&var4 = std::move(var1);
char1 var1;
char1 *var2;
char1 &var3 = var1;
char1 &&var4 = std::move(var1);
}

{
// CHECK: uint8_t var1;
// CHECK-NEXT: uint8_t *var2;
// CHECK-NEXT: uint8_t &var3 = var1;
// CHECK-NEXT: uint8_t &&var4 = std::move(var1);
uchar1 var1;
uchar1 *var2;
uchar1 &var3 = var1;
uchar1 &&var4 = std::move(var1);
}

{
// CHECK: sycl::char2 var1;
// CHECK-NEXT: sycl::char2 *var2;
// CHECK-NEXT: sycl::char2 &var3 = var1;
// CHECK-NEXT: sycl::char2 &&var4 = std::move(var1);
char2 var1;
char2 *var2;
char2 &var3 = var1;
char2 &&var4 = std::move(var1);
}

{
// CHECK: sycl::uchar2 var1;
// CHECK-NEXT: sycl::uchar2 *var2;
// CHECK-NEXT: sycl::uchar2 &var3 = var1;
// CHECK-NEXT: sycl::uchar2 &&var4 = std::move(var1);
uchar2 var1;
uchar2 *var2;
uchar2 &var3 = var1;
uchar2 &&var4 = std::move(var1);
}

{
// CHECK: sycl::char3 var1;
// CHECK-NEXT: sycl::char3 *var2;
// CHECK-NEXT: sycl::char3 &var3 = var1;
// CHECK-NEXT: sycl::char3 &&var4 = std::move(var1);
char3 var1;
char3 *var2;
char3 &var3 = var1;
char3 &&var4 = std::move(var1);
}

{
// CHECK: sycl::uchar3 var1;
// CHECK-NEXT: sycl::uchar3 *var2;
// CHECK-NEXT: sycl::uchar3 &var3 = var1;
// CHECK-NEXT: sycl::uchar3 &&var4 = std::move(var1);
uchar3 var1;
uchar3 *var2;
uchar3 &var3 = var1;
uchar3 &&var4 = std::move(var1);
}

{
// CHECK: sycl::char4 var1;
// CHECK-NEXT: sycl::char4 *var2;
// CHECK-NEXT: sycl::char4 &var3 = var1;
// CHECK-NEXT: sycl::char4 &&var4 = std::move(var1);
char4 var1;
char4 *var2;
char4 &var3 = var1;
char4 &&var4 = std::move(var1);
}

{
// CHECK: sycl::uchar4 var1;
// CHECK-NEXT: sycl::uchar4 *var2;
// CHECK-NEXT: sycl::uchar4 &var3 = var1;
// CHECK-NEXT: sycl::uchar4 &&var4 = std::move(var1);
uchar4 var1;
uchar4 *var2;
uchar4 &var3 = var1;
uchar4 &&var4 = std::move(var1);
}

{
// CHECK: short var1;
// CHECK-NEXT: short *var2;
// CHECK-NEXT: short &var3 = var1;
// CHECK-NEXT: short &&var4 = std::move(var1);
short1 var1;
short1 *var2;
short1 &var3 = var1;
short1 &&var4 = std::move(var1);
}

{
// CHECK: uint16_t var1;
// CHECK-NEXT: uint16_t *var2;
// CHECK-NEXT: uint16_t &var3 = var1;
// CHECK-NEXT: uint16_t &&var4 = std::move(var1);
ushort1 var1;
ushort1 *var2;
ushort1 &var3 = var1;
ushort1 &&var4 = std::move(var1);
}

{
// CHECK: sycl::short2 var1;
// CHECK-NEXT: sycl::short2 *var2;
// CHECK-NEXT: sycl::short2 &var3 = var1;
// CHECK-NEXT: sycl::short2 &&var4 = std::move(var1);
short2 var1;
short2 *var2;
short2 &var3 = var1;
short2 &&var4 = std::move(var1);
}

{
// CHECK: sycl::ushort2 var1;
// CHECK-NEXT: sycl::ushort2 *var2;
// CHECK-NEXT: sycl::ushort2 &var3 = var1;
// CHECK-NEXT: sycl::ushort2 &&var4 = std::move(var1);
ushort2 var1;
ushort2 *var2;
ushort2 &var3 = var1;
ushort2 &&var4 = std::move(var1);
}

{
// CHECK: sycl::short3 var1;
// CHECK-NEXT: sycl::short3 *var2;
// CHECK-NEXT: sycl::short3 &var3 = var1;
// CHECK-NEXT: sycl::short3 &&var4 = std::move(var1);
short3 var1;
short3 *var2;
short3 &var3 = var1;
short3 &&var4 = std::move(var1);
}

{
// CHECK: sycl::ushort3 var1;
// CHECK-NEXT: sycl::ushort3 *var2;
// CHECK-NEXT: sycl::ushort3 &var3 = var1;
// CHECK-NEXT: sycl::ushort3 &&var4 = std::move(var1);
ushort3 var1;
ushort3 *var2;
ushort3 &var3 = var1;
ushort3 &&var4 = std::move(var1);
}

{
// CHECK: sycl::short4 var1;
// CHECK-NEXT: sycl::short4 *var2;
// CHECK-NEXT: sycl::short4 &var3 = var1;
// CHECK-NEXT: sycl::short4 &&var4 = std::move(var1);
short4 var1;
short4 *var2;
short4 &var3 = var1;
short4 &&var4 = std::move(var1);
}

{
// CHECK: sycl::ushort4 var1;
// CHECK-NEXT: sycl::ushort4 *var2;
// CHECK-NEXT: sycl::ushort4 &var3 = var1;
// CHECK-NEXT: sycl::ushort4 &&var4 = std::move(var1);
ushort4 var1;
ushort4 *var2;
ushort4 &var3 = var1;
ushort4 &&var4 = std::move(var1);
}

{
// CHECK: int var1;
// CHECK-NEXT: int *var2;
// CHECK-NEXT: int &var3 = var1;
// CHECK-NEXT: int &&var4 = std::move(var1);
int1 var1;
int1 *var2;
int1 &var3 = var1;
int1 &&var4 = std::move(var1);
}

{
// CHECK: uint32_t var1;
// CHECK-NEXT: uint32_t *var2;
// CHECK-NEXT: uint32_t &var3 = var1;
// CHECK-NEXT: uint32_t &&var4 = std::move(var1);
uint1 var1;
uint1 *var2;
uint1 &var3 = var1;
uint1 &&var4 = std::move(var1);
}

{
// CHECK: sycl::int2 var1;
// CHECK-NEXT: sycl::int2 *var2;
// CHECK-NEXT: sycl::int2 &var3 = var1;
// CHECK-NEXT: sycl::int2 &&var4 = std::move(var1);
int2 var1;
int2 *var2;
int2 &var3 = var1;
int2 &&var4 = std::move(var1);
}

{
// CHECK: sycl::uint2 var1;
// CHECK-NEXT: sycl::uint2 *var2;
// CHECK-NEXT: sycl::uint2 &var3 = var1;
// CHECK-NEXT: sycl::uint2 &&var4 = std::move(var1);
uint2 var1;
uint2 *var2;
uint2 &var3 = var1;
uint2 &&var4 = std::move(var1);
}

{
// CHECK: sycl::int3 var1;
// CHECK-NEXT: sycl::int3 *var2;
// CHECK-NEXT: sycl::int3 &var3 = var1;
// CHECK-NEXT: sycl::int3 &&var4 = std::move(var1);
int3 var1;
int3 *var2;
int3 &var3 = var1;
int3 &&var4 = std::move(var1);
}

{
// CHECK: sycl::uint3 var1;
// CHECK-NEXT: sycl::uint3 *var2;
// CHECK-NEXT: sycl::uint3 &var3 = var1;
// CHECK-NEXT: sycl::uint3 &&var4 = std::move(var1);
uint3 var1;
uint3 *var2;
uint3 &var3 = var1;
uint3 &&var4 = std::move(var1);
}

{
// CHECK: sycl::int4 var1;
// CHECK-NEXT: sycl::int4 *var2;
// CHECK-NEXT: sycl::int4 &var3 = var1;
// CHECK-NEXT: sycl::int4 &&var4 = std::move(var1);
int4 var1;
int4 *var2;
int4 &var3 = var1;
int4 &&var4 = std::move(var1);
}

{
// CHECK: sycl::uint4 var1;
// CHECK-NEXT: sycl::uint4 *var2;
// CHECK-NEXT: sycl::uint4 &var3 = var1;
// CHECK-NEXT: sycl::uint4 &&var4 = std::move(var1);
uint4 var1;
uint4 *var2;
uint4 &var3 = var1;
uint4 &&var4 = std::move(var1);
}

{
// CHECK: long var1;
// CHECK-NEXT: long *var2;
// CHECK-NEXT: long &var3 = var1;
// CHECK-NEXT: long &&var4 = std::move(var1);
long1 var1;
long1 *var2;
long1 &var3 = var1;
long1 &&var4 = std::move(var1);
}

{
// CHECK: uint64_t var1;
// CHECK-NEXT: uint64_t *var2;
// CHECK-NEXT: uint64_t &var3 = var1;
// CHECK-NEXT: uint64_t &&var4 = std::move(var1);
ulong1 var1;
ulong1 *var2;
ulong1 &var3 = var1;
ulong1 &&var4 = std::move(var1);
}

{
// CHECK: sycl::long2 var1;
// CHECK-NEXT: sycl::long2 *var2;
// CHECK-NEXT: sycl::long2 &var3 = var1;
// CHECK-NEXT: sycl::long2 &&var4 = std::move(var1);
long2 var1;
long2 *var2;
long2 &var3 = var1;
long2 &&var4 = std::move(var1);
}

{
// CHECK: sycl::ulong2 var1;
// CHECK-NEXT: sycl::ulong2 *var2;
// CHECK-NEXT: sycl::ulong2 &var3 = var1;
// CHECK-NEXT: sycl::ulong2 &&var4 = std::move(var1);
ulong2 var1;
ulong2 *var2;
ulong2 &var3 = var1;
ulong2 &&var4 = std::move(var1);
}

{
// CHECK: sycl::long3 var1;
// CHECK-NEXT: sycl::long3 *var2;
// CHECK-NEXT: sycl::long3 &var3 = var1;
// CHECK-NEXT: sycl::long3 &&var4 = std::move(var1);
long3 var1;
long3 *var2;
long3 &var3 = var1;
long3 &&var4 = std::move(var1);
}

{
// CHECK: sycl::ulong3 var1;
// CHECK-NEXT: sycl::ulong3 *var2;
// CHECK-NEXT: sycl::ulong3 &var3 = var1;
// CHECK-NEXT: sycl::ulong3 &&var4 = std::move(var1);
ulong3 var1;
ulong3 *var2;
ulong3 &var3 = var1;
ulong3 &&var4 = std::move(var1);
}

{
// CHECK: sycl::long4 var1;
// CHECK-NEXT: sycl::long4 *var2;
// CHECK-NEXT: sycl::long4 &var3 = var1;
// CHECK-NEXT: sycl::long4 &&var4 = std::move(var1);
long4 var1;
long4 *var2;
long4 &var3 = var1;
long4 &&var4 = std::move(var1);
}

{
// CHECK: sycl::ulong4 var1;
// CHECK-NEXT: sycl::ulong4 *var2;
// CHECK-NEXT: sycl::ulong4 &var3 = var1;
// CHECK-NEXT: sycl::ulong4 &&var4 = std::move(var1);
ulong4 var1;
ulong4 *var2;
ulong4 &var3 = var1;
ulong4 &&var4 = std::move(var1);
}

{
// CHECK: float var1;
// CHECK-NEXT: float *var2;
// CHECK-NEXT: float &var3 = var1;
// CHECK-NEXT: float &&var4 = std::move(var1);
float1 var1;
float1 *var2;
float1 &var3 = var1;
float1 &&var4 = std::move(var1);
}

{
// CHECK: sycl::float2 var1;
// CHECK-NEXT: sycl::float2 *var2;
// CHECK-NEXT: sycl::float2 &var3 = var1;
// CHECK-NEXT: sycl::float2 &&var4 = std::move(var1);
float2 var1;
float2 *var2;
float2 &var3 = var1;
float2 &&var4 = std::move(var1);
}

{
// CHECK: sycl::float3 var1;
// CHECK-NEXT: sycl::float3 *var2;
// CHECK-NEXT: sycl::float3 &var3 = var1;
// CHECK-NEXT: sycl::float3 &&var4 = std::move(var1);
float3 var1;
float3 *var2;
float3 &var3 = var1;
float3 &&var4 = std::move(var1);
}

{
// CHECK: sycl::float4 var1;
// CHECK-NEXT: sycl::float4 *var2;
// CHECK-NEXT: sycl::float4 &var3 = var1;
// CHECK-NEXT: sycl::float4 &&var4 = std::move(var1);
float4 var1;
float4 *var2;
float4 &var3 = var1;
float4 &&var4 = std::move(var1);
}

{
// CHECK: int64_t var1;
// CHECK-NEXT: int64_t *var2;
// CHECK-NEXT: int64_t &var3 = var1;
// CHECK-NEXT: int64_t &&var4 = std::move(var1);
longlong1 var1;
longlong1 *var2;
longlong1 &var3 = var1;
longlong1 &&var4 = std::move(var1);
}

{
// CHECK: uint64_t var1;
// CHECK-NEXT: uint64_t *var2;
// CHECK-NEXT: uint64_t &var3 = var1;
// CHECK-NEXT: uint64_t &&var4 = std::move(var1);
ulonglong1 var1;
ulonglong1 *var2;
ulonglong1 &var3 = var1;
ulonglong1 &&var4 = std::move(var1);
}

{
// CHECK: sycl::longlong2 var1;
// CHECK-NEXT: sycl::longlong2 *var2;
// CHECK-NEXT: sycl::longlong2 &var3 = var1;
// CHECK-NEXT: sycl::longlong2 &&var4 = std::move(var1);
longlong2 var1;
longlong2 *var2;
longlong2 &var3 = var1;
longlong2 &&var4 = std::move(var1);
}

{
// CHECK: sycl::ulonglong2 var1;
// CHECK-NEXT: sycl::ulonglong2 *var2;
// CHECK-NEXT: sycl::ulonglong2 &var3 = var1;
// CHECK-NEXT: sycl::ulonglong2 &&var4 = std::move(var1);
ulonglong2 var1;
ulonglong2 *var2;
ulonglong2 &var3 = var1;
ulonglong2 &&var4 = std::move(var1);
}

{
// CHECK: sycl::longlong3 var1;
// CHECK-NEXT: sycl::longlong3 *var2;
// CHECK-NEXT: sycl::longlong3 &var3 = var1;
// CHECK-NEXT: sycl::longlong3 &&var4 = std::move(var1);
longlong3 var1;
longlong3 *var2;
longlong3 &var3 = var1;
longlong3 &&var4 = std::move(var1);
}

{
// CHECK: sycl::ulonglong3 var1;
// CHECK-NEXT: sycl::ulonglong3 *var2;
// CHECK-NEXT: sycl::ulonglong3 &var3 = var1;
// CHECK-NEXT: sycl::ulonglong3 &&var4 = std::move(var1);
ulonglong3 var1;
ulonglong3 *var2;
ulonglong3 &var3 = var1;
ulonglong3 &&var4 = std::move(var1);
}

{
// CHECK: sycl::longlong4 var1;
// CHECK-NEXT: sycl::longlong4 *var2;
// CHECK-NEXT: sycl::longlong4 &var3 = var1;
// CHECK-NEXT: sycl::longlong4 &&var4 = std::move(var1);
longlong4 var1;
longlong4 *var2;
longlong4 &var3 = var1;
longlong4 &&var4 = std::move(var1);
}

{
// CHECK: sycl::ulonglong4 var1;
// CHECK-NEXT: sycl::ulonglong4 *var2;
// CHECK-NEXT: sycl::ulonglong4 &var3 = var1;
// CHECK-NEXT: sycl::ulonglong4 &&var4 = std::move(var1);
ulonglong4 var1;
ulonglong4 *var2;
ulonglong4 &var3 = var1;
ulonglong4 &&var4 = std::move(var1);
}

{
// CHECK: double var1;
// CHECK-NEXT: double *var2;
// CHECK-NEXT: double &var3 = var1;
// CHECK-NEXT: double &&var4 = std::move(var1);
double1 var1;
double1 *var2;
double1 &var3 = var1;
double1 &&var4 = std::move(var1);
}

{
// CHECK: sycl::double2 var1;
// CHECK-NEXT: sycl::double2 *var2;
// CHECK-NEXT: sycl::double2 &var3 = var1;
// CHECK-NEXT: sycl::double2 &&var4 = std::move(var1);
double2 var1;
double2 *var2;
double2 &var3 = var1;
double2 &&var4 = std::move(var1);
}

{
// CHECK: sycl::double3 var1;
// CHECK-NEXT: sycl::double3 *var2;
// CHECK-NEXT: sycl::double3 &var3 = var1;
// CHECK-NEXT: sycl::double3 &&var4 = std::move(var1);
double3 var1;
double3 *var2;
double3 &var3 = var1;
double3 &&var4 = std::move(var1);
}

{
// CHECK: sycl::double4 var1;
// CHECK-NEXT: sycl::double4 *var2;
// CHECK-NEXT: sycl::double4 &var3 = var1;
// CHECK-NEXT: sycl::double4 &&var4 = std::move(var1);
double4 var1;
double4 *var2;
double4 &var3 = var1;
double4 &&var4 = std::move(var1);
}
}

void case_2(void) {
{
// CHECK:  new char();
// CHECK-NEXT:  new char *();
  new char1();
  new char1 *();
}

{
// CHECK:  new uint8_t();
// CHECK-NEXT:  new uint8_t *();
  new uchar1();
  new uchar1 *();
}

{
// CHECK:  new sycl::char2();
// CHECK-NEXT:  new sycl::char2 *();
  new char2();
  new char2 *();
}

{
// CHECK:  new sycl::uchar2();
// CHECK-NEXT:  new sycl::uchar2 *();
  new uchar2();
  new uchar2 *();
}

{
// CHECK:  new sycl::char3();
// CHECK-NEXT:  new sycl::char3 *();
  new char3();
  new char3 *();
}

{
// CHECK:  new sycl::uchar3();
// CHECK-NEXT:  new sycl::uchar3 *();
  new uchar3();
  new uchar3 *();
}

{
// CHECK:  new sycl::char4();
// CHECK-NEXT:  new sycl::char4 *();
  new char4();
  new char4 *();
}

{
// CHECK:  new sycl::uchar4();
// CHECK-NEXT:  new sycl::uchar4 *();
  new uchar4();
  new uchar4 *();
}

{
// CHECK:  new short();
// CHECK-NEXT:  new short *();
  new short1();
  new short1 *();
}

{
// CHECK:  new uint16_t();
// CHECK-NEXT:  new uint16_t *();
  new ushort1();
  new ushort1 *();
}

{
// CHECK:  new sycl::short2();
// CHECK-NEXT:  new sycl::short2 *();
  new short2();
  new short2 *();
}

{
// CHECK:  new sycl::ushort2();
// CHECK-NEXT:  new sycl::ushort2 *();
  new ushort2();
  new ushort2 *();
}

{
// CHECK:  new sycl::short3();
// CHECK-NEXT:  new sycl::short3 *();
  new short3();
  new short3 *();
}

{
// CHECK:  new sycl::ushort3();
// CHECK-NEXT:  new sycl::ushort3 *();
  new ushort3();
  new ushort3 *();
}

{
// CHECK:  new sycl::short4();
// CHECK-NEXT:  new sycl::short4 *();
  new short4();
  new short4 *();
}

{
// CHECK:  new sycl::ushort4();
// CHECK-NEXT:  new sycl::ushort4 *();
  new ushort4();
  new ushort4 *();
}

{
// CHECK:  new int();
// CHECK-NEXT:  new int *();
  new int1();
  new int1 *();
}

{
// CHECK:  new uint32_t();
// CHECK-NEXT:  new uint32_t *();
  new uint1();
  new uint1 *();
}

{
// CHECK:  new sycl::int2();
// CHECK-NEXT:  new sycl::int2 *();
  new int2();
  new int2 *();
}

{
// CHECK:  new sycl::uint2();
// CHECK-NEXT:  new sycl::uint2 *();
  new uint2();
  new uint2 *();
}

{
// CHECK:  new sycl::int3();
// CHECK-NEXT:  new sycl::int3 *();
  new int3();
  new int3 *();
}

{
// CHECK:  new sycl::uint3();
// CHECK-NEXT:  new sycl::uint3 *();
  new uint3();
  new uint3 *();
}

{
// CHECK:  new sycl::int4();
// CHECK-NEXT:  new sycl::int4 *();
  new int4();
  new int4 *();
}

{
// CHECK:  new sycl::uint4();
// CHECK-NEXT:  new sycl::uint4 *();
  new uint4();
  new uint4 *();
}

{
// CHECK:  new long();
// CHECK-NEXT:  new long *();
  new long1();
  new long1 *();
}

{
// CHECK:  new uint64_t();
// CHECK-NEXT:  new uint64_t *();
  new ulong1();
  new ulong1 *();
}

{
// CHECK:  new sycl::long2();
// CHECK-NEXT:  new sycl::long2 *();
  new long2();
  new long2 *();
}

{
// CHECK:  new sycl::ulong2();
// CHECK-NEXT:  new sycl::ulong2 *();
  new ulong2();
  new ulong2 *();
}

{
// CHECK:  new sycl::long3();
// CHECK-NEXT:  new sycl::long3 *();
  new long3();
  new long3 *();
}

{
// CHECK:  new sycl::ulong3();
// CHECK-NEXT:  new sycl::ulong3 *();
  new ulong3();
  new ulong3 *();
}

{
// CHECK:  new sycl::long4();
// CHECK-NEXT:  new sycl::long4 *();
  new long4();
  new long4 *();
}

{
// CHECK:  new sycl::ulong4();
// CHECK-NEXT:  new sycl::ulong4 *();
  new ulong4();
  new ulong4 *();
}

{
// CHECK:  new float();
// CHECK-NEXT:  new float *();
  new float1();
  new float1 *();
}

{
// CHECK:  new sycl::float2();
// CHECK-NEXT:  new sycl::float2 *();
  new float2();
  new float2 *();
}

{
// CHECK:  new sycl::float3();
// CHECK-NEXT:  new sycl::float3 *();
  new float3();
  new float3 *();
}

{
// CHECK:  new sycl::float4();
// CHECK-NEXT:  new sycl::float4 *();
  new float4();
  new float4 *();
}

{
// CHECK:  new int64_t();
// CHECK-NEXT:  new int64_t *();
  new longlong1();
  new longlong1 *();
}

{
// CHECK:  new uint64_t();
// CHECK-NEXT:  new uint64_t *();
  new ulonglong1();
  new ulonglong1 *();
}

{
// CHECK:  new sycl::longlong2();
// CHECK-NEXT:  new sycl::longlong2 *();
  new longlong2();
  new longlong2 *();
}

{
// CHECK:  new sycl::ulonglong2();
// CHECK-NEXT:  new sycl::ulonglong2 *();
  new ulonglong2();
  new ulonglong2 *();
}

{
// CHECK:  new sycl::longlong3();
// CHECK-NEXT:  new sycl::longlong3 *();
  new longlong3();
  new longlong3 *();
}

{
// CHECK:  new sycl::ulonglong3();
// CHECK-NEXT:  new sycl::ulonglong3 *();
  new ulonglong3();
  new ulonglong3 *();
}

{
// CHECK:  new sycl::longlong4();
// CHECK-NEXT:  new sycl::longlong4 *();
  new longlong4();
  new longlong4 *();
}

{
// CHECK:  new sycl::ulonglong4();
// CHECK-NEXT:  new sycl::ulonglong4 *();
  new ulonglong4();
  new ulonglong4 *();
}

{
// CHECK:  new double();
// CHECK-NEXT:  new double *();
  new double1();
  new double1 *();
}

{
// CHECK:  new sycl::double2();
// CHECK-NEXT:  new sycl::double2 *();
  new double2();
  new double2 *();
}

{
// CHECK:  new sycl::double3();
// CHECK-NEXT:  new sycl::double3 *();
  new double3();
  new double3 *();
}

{
// CHECK:  new sycl::double4();
// CHECK-NEXT:  new sycl::double4 *();
  new double4();
  new double4 *();
}
}


// case 3
// CHECK: char foo0();
// CHECK-NEXT: char *foo1();
// CHECK-NEXT: char &foo2();
char1 foo0();
char1 *foo1();
char1 &foo2();

// CHECK: uint8_t foo3();
// CHECK-NEXT: uint8_t *foo4();
// CHECK-NEXT: uint8_t &foo5();
uchar1 foo3();
uchar1 *foo4();
uchar1 &foo5();

// CHECK: sycl::char2 foo6();
// CHECK-NEXT: sycl::char2 *foo7();
// CHECK-NEXT: sycl::char2 &foo8();
char2 foo6();
char2 *foo7();
char2 &foo8();

// CHECK: sycl::uchar2 foo9();
// CHECK-NEXT: sycl::uchar2 *foo10();
// CHECK-NEXT: sycl::uchar2 &foo11();
uchar2 foo9();
uchar2 *foo10();
uchar2 &foo11();

// CHECK: sycl::char3 foo12();
// CHECK-NEXT: sycl::char3 *foo13();
// CHECK-NEXT: sycl::char3 &foo14();
char3 foo12();
char3 *foo13();
char3 &foo14();

// CHECK: sycl::uchar3 foo15();
// CHECK-NEXT: sycl::uchar3 *foo16();
// CHECK-NEXT: sycl::uchar3 &foo17();
uchar3 foo15();
uchar3 *foo16();
uchar3 &foo17();

// CHECK: sycl::char4 foo18();
// CHECK-NEXT: sycl::char4 *foo19();
// CHECK-NEXT: sycl::char4 &foo20();
char4 foo18();
char4 *foo19();
char4 &foo20();

// CHECK: sycl::uchar4 foo21();
// CHECK-NEXT: sycl::uchar4 *foo22();
// CHECK-NEXT: sycl::uchar4 &foo23();
uchar4 foo21();
uchar4 *foo22();
uchar4 &foo23();

// CHECK: short foo24();
// CHECK-NEXT: short *foo25();
// CHECK-NEXT: short &foo26();
short1 foo24();
short1 *foo25();
short1 &foo26();

// CHECK: uint16_t foo27();
// CHECK-NEXT: uint16_t *foo28();
// CHECK-NEXT: uint16_t &foo29();
ushort1 foo27();
ushort1 *foo28();
ushort1 &foo29();

// CHECK: sycl::short2 foo30();
// CHECK-NEXT: sycl::short2 *foo31();
// CHECK-NEXT: sycl::short2 &foo32();
short2 foo30();
short2 *foo31();
short2 &foo32();

// CHECK: sycl::ushort2 foo33();
// CHECK-NEXT: sycl::ushort2 *foo34();
// CHECK-NEXT: sycl::ushort2 &foo35();
ushort2 foo33();
ushort2 *foo34();
ushort2 &foo35();

// CHECK: sycl::short3 foo36();
// CHECK-NEXT: sycl::short3 *foo37();
// CHECK-NEXT: sycl::short3 &foo38();
short3 foo36();
short3 *foo37();
short3 &foo38();

// CHECK: sycl::ushort3 foo39();
// CHECK-NEXT: sycl::ushort3 *foo40();
// CHECK-NEXT: sycl::ushort3 &foo41();
ushort3 foo39();
ushort3 *foo40();
ushort3 &foo41();

// CHECK: sycl::short4 foo42();
// CHECK-NEXT: sycl::short4 *foo43();
// CHECK-NEXT: sycl::short4 &foo44();
short4 foo42();
short4 *foo43();
short4 &foo44();

// CHECK: sycl::ushort4 foo45();
// CHECK-NEXT: sycl::ushort4 *foo46();
// CHECK-NEXT: sycl::ushort4 &foo47();
ushort4 foo45();
ushort4 *foo46();
ushort4 &foo47();

// CHECK: int foo48();
// CHECK-NEXT: int *foo49();
// CHECK-NEXT: int &foo50();
int1 foo48();
int1 *foo49();
int1 &foo50();

// CHECK: uint32_t foo51();
// CHECK-NEXT: uint32_t *foo52();
// CHECK-NEXT: uint32_t &foo53();
uint1 foo51();
uint1 *foo52();
uint1 &foo53();

// CHECK: sycl::int2 foo54();
// CHECK-NEXT: sycl::int2 *foo55();
// CHECK-NEXT: sycl::int2 &foo56();
int2 foo54();
int2 *foo55();
int2 &foo56();

// CHECK: sycl::uint2 foo57();
// CHECK-NEXT: sycl::uint2 *foo58();
// CHECK-NEXT: sycl::uint2 &foo59();
uint2 foo57();
uint2 *foo58();
uint2 &foo59();

// CHECK: sycl::int3 foo60();
// CHECK-NEXT: sycl::int3 *foo61();
// CHECK-NEXT: sycl::int3 &foo62();
int3 foo60();
int3 *foo61();
int3 &foo62();

// CHECK: sycl::uint3 foo63();
// CHECK-NEXT: sycl::uint3 *foo64();
// CHECK-NEXT: sycl::uint3 &foo65();
uint3 foo63();
uint3 *foo64();
uint3 &foo65();

// CHECK: sycl::int4 foo66();
// CHECK-NEXT: sycl::int4 *foo67();
// CHECK-NEXT: sycl::int4 &foo68();
int4 foo66();
int4 *foo67();
int4 &foo68();

// CHECK: sycl::uint4 foo69();
// CHECK-NEXT: sycl::uint4 *foo70();
// CHECK-NEXT: sycl::uint4 &foo71();
uint4 foo69();
uint4 *foo70();
uint4 &foo71();

// CHECK: long foo72();
// CHECK-NEXT: long *foo73();
// CHECK-NEXT: long &foo74();
long1 foo72();
long1 *foo73();
long1 &foo74();

// CHECK: uint64_t foo75();
// CHECK-NEXT: uint64_t *foo76();
// CHECK-NEXT: uint64_t &foo77();
ulong1 foo75();
ulong1 *foo76();
ulong1 &foo77();

// CHECK: sycl::long2 foo78();
// CHECK-NEXT: sycl::long2 *foo79();
// CHECK-NEXT: sycl::long2 &foo80();
long2 foo78();
long2 *foo79();
long2 &foo80();

// CHECK: sycl::ulong2 foo81();
// CHECK-NEXT: sycl::ulong2 *foo82();
// CHECK-NEXT: sycl::ulong2 &foo83();
ulong2 foo81();
ulong2 *foo82();
ulong2 &foo83();

// CHECK: sycl::long3 foo84();
// CHECK-NEXT: sycl::long3 *foo85();
// CHECK-NEXT: sycl::long3 &foo86();
long3 foo84();
long3 *foo85();
long3 &foo86();

// CHECK: sycl::ulong3 foo87();
// CHECK-NEXT: sycl::ulong3 *foo88();
// CHECK-NEXT: sycl::ulong3 &foo89();
ulong3 foo87();
ulong3 *foo88();
ulong3 &foo89();

// CHECK: sycl::long4 foo90();
// CHECK-NEXT: sycl::long4 *foo91();
// CHECK-NEXT: sycl::long4 &foo92();
long4 foo90();
long4 *foo91();
long4 &foo92();

// CHECK: sycl::ulong4 foo93();
// CHECK-NEXT: sycl::ulong4 *foo94();
// CHECK-NEXT: sycl::ulong4 &foo95();
ulong4 foo93();
ulong4 *foo94();
ulong4 &foo95();

// CHECK: float foo96();
// CHECK-NEXT: float *foo97();
// CHECK-NEXT: float &foo98();
float1 foo96();
float1 *foo97();
float1 &foo98();

// CHECK: sycl::float2 foo99();
// CHECK-NEXT: sycl::float2 *foo100();
// CHECK-NEXT: sycl::float2 &foo101();
float2 foo99();
float2 *foo100();
float2 &foo101();

// CHECK: sycl::float3 foo102();
// CHECK-NEXT: sycl::float3 *foo103();
// CHECK-NEXT: sycl::float3 &foo104();
float3 foo102();
float3 *foo103();
float3 &foo104();

// CHECK: sycl::float4 foo105();
// CHECK-NEXT: sycl::float4 *foo106();
// CHECK-NEXT: sycl::float4 &foo107();
float4 foo105();
float4 *foo106();
float4 &foo107();

// CHECK: int64_t foo108();
// CHECK-NEXT: int64_t *foo109();
// CHECK-NEXT: int64_t &foo110();
longlong1 foo108();
longlong1 *foo109();
longlong1 &foo110();

// CHECK: uint64_t foo111();
// CHECK-NEXT: uint64_t *foo112();
// CHECK-NEXT: uint64_t &foo113();
ulonglong1 foo111();
ulonglong1 *foo112();
ulonglong1 &foo113();

// CHECK: sycl::longlong2 foo114();
// CHECK-NEXT: sycl::longlong2 *foo115();
// CHECK-NEXT: sycl::longlong2 &foo116();
longlong2 foo114();
longlong2 *foo115();
longlong2 &foo116();

// CHECK: sycl::ulonglong2 foo117();
// CHECK-NEXT: sycl::ulonglong2 *foo118();
// CHECK-NEXT: sycl::ulonglong2 &foo119();
ulonglong2 foo117();
ulonglong2 *foo118();
ulonglong2 &foo119();

// CHECK: sycl::longlong3 foo120();
// CHECK-NEXT: sycl::longlong3 *foo121();
// CHECK-NEXT: sycl::longlong3 &foo122();
longlong3 foo120();
longlong3 *foo121();
longlong3 &foo122();

// CHECK: sycl::ulonglong3 foo123();
// CHECK-NEXT: sycl::ulonglong3 *foo124();
// CHECK-NEXT: sycl::ulonglong3 &foo125();
ulonglong3 foo123();
ulonglong3 *foo124();
ulonglong3 &foo125();

// CHECK: sycl::longlong4 foo126();
// CHECK-NEXT: sycl::longlong4 *foo127();
// CHECK-NEXT: sycl::longlong4 &foo128();
longlong4 foo126();
longlong4 *foo127();
longlong4 &foo128();

// CHECK: sycl::ulonglong4 foo129();
// CHECK-NEXT: sycl::ulonglong4 *foo130();
// CHECK-NEXT: sycl::ulonglong4 &foo131();
ulonglong4 foo129();
ulonglong4 *foo130();
ulonglong4 &foo131();

// CHECK: double foo132();
// CHECK-NEXT: double *foo133();
// CHECK-NEXT: double &foo134();
double1 foo132();
double1 *foo133();
double1 &foo134();

// CHECK: sycl::double2 foo135();
// CHECK-NEXT: sycl::double2 *foo136();
// CHECK-NEXT: sycl::double2 &foo137();
double2 foo135();
double2 *foo136();
double2 &foo137();

// CHECK: sycl::double3 foo138();
// CHECK-NEXT: sycl::double3 *foo139();
// CHECK-NEXT: sycl::double3 &foo140();
double3 foo138();
double3 *foo139();
double3 &foo140();

// CHECK: sycl::double4 foo141();
// CHECK-NEXT: sycl::double4 *foo142();
// CHECK-NEXT: sycl::double4 &foo143();
double4 foo141();
double4 *foo142();
double4 &foo143();


// case 4
template <typename T> struct S {};
// CHECK: template <> struct S<char> {};
// CHECK-NEXT: template <> struct S<char *> {};
// CHECK-NEXT: template <> struct S<char &> {};
// CHECK-NEXT: template <> struct S<char &&> {};
template <> struct S<char1> {};
template <> struct S<char1 *> {};
template <> struct S<char1 &> {};
template <> struct S<char1 &&> {};

// CHECK: template <> struct S<uint8_t> {};
// CHECK-NEXT: template <> struct S<uint8_t *> {};
// CHECK-NEXT: template <> struct S<uint8_t &> {};
// CHECK-NEXT: template <> struct S<uint8_t &&> {};
template <> struct S<uchar1> {};
template <> struct S<uchar1 *> {};
template <> struct S<uchar1 &> {};
template <> struct S<uchar1 &&> {};

// CHECK: template <> struct S<sycl::char2> {};
// CHECK-NEXT: template <> struct S<sycl::char2 *> {};
// CHECK-NEXT: template <> struct S<sycl::char2 &> {};
// CHECK-NEXT: template <> struct S<sycl::char2 &&> {};
template <> struct S<char2> {};
template <> struct S<char2 *> {};
template <> struct S<char2 &> {};
template <> struct S<char2 &&> {};

// CHECK: template <> struct S<sycl::uchar2> {};
// CHECK-NEXT: template <> struct S<sycl::uchar2 *> {};
// CHECK-NEXT: template <> struct S<sycl::uchar2 &> {};
// CHECK-NEXT: template <> struct S<sycl::uchar2 &&> {};
template <> struct S<uchar2> {};
template <> struct S<uchar2 *> {};
template <> struct S<uchar2 &> {};
template <> struct S<uchar2 &&> {};

// CHECK: template <> struct S<sycl::char3> {};
// CHECK-NEXT: template <> struct S<sycl::char3 *> {};
// CHECK-NEXT: template <> struct S<sycl::char3 &> {};
// CHECK-NEXT: template <> struct S<sycl::char3 &&> {};
template <> struct S<char3> {};
template <> struct S<char3 *> {};
template <> struct S<char3 &> {};
template <> struct S<char3 &&> {};

// CHECK: template <> struct S<sycl::uchar3> {};
// CHECK-NEXT: template <> struct S<sycl::uchar3 *> {};
// CHECK-NEXT: template <> struct S<sycl::uchar3 &> {};
// CHECK-NEXT: template <> struct S<sycl::uchar3 &&> {};
template <> struct S<uchar3> {};
template <> struct S<uchar3 *> {};
template <> struct S<uchar3 &> {};
template <> struct S<uchar3 &&> {};

// CHECK: template <> struct S<sycl::char4> {};
// CHECK-NEXT: template <> struct S<sycl::char4 *> {};
// CHECK-NEXT: template <> struct S<sycl::char4 &> {};
// CHECK-NEXT: template <> struct S<sycl::char4 &&> {};
template <> struct S<char4> {};
template <> struct S<char4 *> {};
template <> struct S<char4 &> {};
template <> struct S<char4 &&> {};

// CHECK: template <> struct S<sycl::uchar4> {};
// CHECK-NEXT: template <> struct S<sycl::uchar4 *> {};
// CHECK-NEXT: template <> struct S<sycl::uchar4 &> {};
// CHECK-NEXT: template <> struct S<sycl::uchar4 &&> {};
template <> struct S<uchar4> {};
template <> struct S<uchar4 *> {};
template <> struct S<uchar4 &> {};
template <> struct S<uchar4 &&> {};

// CHECK: template <> struct S<short> {};
// CHECK-NEXT: template <> struct S<short *> {};
// CHECK-NEXT: template <> struct S<short &> {};
// CHECK-NEXT: template <> struct S<short &&> {};
template <> struct S<short1> {};
template <> struct S<short1 *> {};
template <> struct S<short1 &> {};
template <> struct S<short1 &&> {};

// CHECK: template <> struct S<uint16_t> {};
// CHECK-NEXT: template <> struct S<uint16_t *> {};
// CHECK-NEXT: template <> struct S<uint16_t &> {};
// CHECK-NEXT: template <> struct S<uint16_t &&> {};
template <> struct S<ushort1> {};
template <> struct S<ushort1 *> {};
template <> struct S<ushort1 &> {};
template <> struct S<ushort1 &&> {};

// CHECK: template <> struct S<sycl::short2> {};
// CHECK-NEXT: template <> struct S<sycl::short2 *> {};
// CHECK-NEXT: template <> struct S<sycl::short2 &> {};
// CHECK-NEXT: template <> struct S<sycl::short2 &&> {};
template <> struct S<short2> {};
template <> struct S<short2 *> {};
template <> struct S<short2 &> {};
template <> struct S<short2 &&> {};

// CHECK: template <> struct S<sycl::ushort2> {};
// CHECK-NEXT: template <> struct S<sycl::ushort2 *> {};
// CHECK-NEXT: template <> struct S<sycl::ushort2 &> {};
// CHECK-NEXT: template <> struct S<sycl::ushort2 &&> {};
template <> struct S<ushort2> {};
template <> struct S<ushort2 *> {};
template <> struct S<ushort2 &> {};
template <> struct S<ushort2 &&> {};

// CHECK: template <> struct S<sycl::short3> {};
// CHECK-NEXT: template <> struct S<sycl::short3 *> {};
// CHECK-NEXT: template <> struct S<sycl::short3 &> {};
// CHECK-NEXT: template <> struct S<sycl::short3 &&> {};
template <> struct S<short3> {};
template <> struct S<short3 *> {};
template <> struct S<short3 &> {};
template <> struct S<short3 &&> {};

// CHECK: template <> struct S<sycl::ushort3> {};
// CHECK-NEXT: template <> struct S<sycl::ushort3 *> {};
// CHECK-NEXT: template <> struct S<sycl::ushort3 &> {};
// CHECK-NEXT: template <> struct S<sycl::ushort3 &&> {};
template <> struct S<ushort3> {};
template <> struct S<ushort3 *> {};
template <> struct S<ushort3 &> {};
template <> struct S<ushort3 &&> {};

// CHECK: template <> struct S<sycl::short4> {};
// CHECK-NEXT: template <> struct S<sycl::short4 *> {};
// CHECK-NEXT: template <> struct S<sycl::short4 &> {};
// CHECK-NEXT: template <> struct S<sycl::short4 &&> {};
template <> struct S<short4> {};
template <> struct S<short4 *> {};
template <> struct S<short4 &> {};
template <> struct S<short4 &&> {};

// CHECK: template <> struct S<sycl::ushort4> {};
// CHECK-NEXT: template <> struct S<sycl::ushort4 *> {};
// CHECK-NEXT: template <> struct S<sycl::ushort4 &> {};
// CHECK-NEXT: template <> struct S<sycl::ushort4 &&> {};
template <> struct S<ushort4> {};
template <> struct S<ushort4 *> {};
template <> struct S<ushort4 &> {};
template <> struct S<ushort4 &&> {};

// CHECK: template <> struct S<int> {};
// CHECK-NEXT: template <> struct S<int *> {};
// CHECK-NEXT: template <> struct S<int &> {};
// CHECK-NEXT: template <> struct S<int &&> {};
template <> struct S<int1> {};
template <> struct S<int1 *> {};
template <> struct S<int1 &> {};
template <> struct S<int1 &&> {};

// CHECK: template <> struct S<uint32_t> {};
// CHECK-NEXT: template <> struct S<uint32_t *> {};
// CHECK-NEXT: template <> struct S<uint32_t &> {};
// CHECK-NEXT: template <> struct S<uint32_t &&> {};
template <> struct S<uint1> {};
template <> struct S<uint1 *> {};
template <> struct S<uint1 &> {};
template <> struct S<uint1 &&> {};

// CHECK: template <> struct S<sycl::int2> {};
// CHECK-NEXT: template <> struct S<sycl::int2 *> {};
// CHECK-NEXT: template <> struct S<sycl::int2 &> {};
// CHECK-NEXT: template <> struct S<sycl::int2 &&> {};
template <> struct S<int2> {};
template <> struct S<int2 *> {};
template <> struct S<int2 &> {};
template <> struct S<int2 &&> {};

// CHECK: template <> struct S<sycl::uint2> {};
// CHECK-NEXT: template <> struct S<sycl::uint2 *> {};
// CHECK-NEXT: template <> struct S<sycl::uint2 &> {};
// CHECK-NEXT: template <> struct S<sycl::uint2 &&> {};
template <> struct S<uint2> {};
template <> struct S<uint2 *> {};
template <> struct S<uint2 &> {};
template <> struct S<uint2 &&> {};

// CHECK: template <> struct S<sycl::int3> {};
// CHECK-NEXT: template <> struct S<sycl::int3 *> {};
// CHECK-NEXT: template <> struct S<sycl::int3 &> {};
// CHECK-NEXT: template <> struct S<sycl::int3 &&> {};
template <> struct S<int3> {};
template <> struct S<int3 *> {};
template <> struct S<int3 &> {};
template <> struct S<int3 &&> {};

// CHECK: template <> struct S<sycl::uint3> {};
// CHECK-NEXT: template <> struct S<sycl::uint3 *> {};
// CHECK-NEXT: template <> struct S<sycl::uint3 &> {};
// CHECK-NEXT: template <> struct S<sycl::uint3 &&> {};
template <> struct S<uint3> {};
template <> struct S<uint3 *> {};
template <> struct S<uint3 &> {};
template <> struct S<uint3 &&> {};

// CHECK: template <> struct S<sycl::int4> {};
// CHECK-NEXT: template <> struct S<sycl::int4 *> {};
// CHECK-NEXT: template <> struct S<sycl::int4 &> {};
// CHECK-NEXT: template <> struct S<sycl::int4 &&> {};
template <> struct S<int4> {};
template <> struct S<int4 *> {};
template <> struct S<int4 &> {};
template <> struct S<int4 &&> {};

// CHECK: template <> struct S<sycl::uint4> {};
// CHECK-NEXT: template <> struct S<sycl::uint4 *> {};
// CHECK-NEXT: template <> struct S<sycl::uint4 &> {};
// CHECK-NEXT: template <> struct S<sycl::uint4 &&> {};
template <> struct S<uint4> {};
template <> struct S<uint4 *> {};
template <> struct S<uint4 &> {};
template <> struct S<uint4 &&> {};

// CHECK: template <> struct S<long> {};
// CHECK-NEXT: template <> struct S<long *> {};
// CHECK-NEXT: template <> struct S<long &> {};
// CHECK-NEXT: template <> struct S<long &&> {};
template <> struct S<long1> {};
template <> struct S<long1 *> {};
template <> struct S<long1 &> {};
template <> struct S<long1 &&> {};

// CHECK: template <> struct S<uint64_t> {};
// CHECK-NEXT: template <> struct S<uint64_t *> {};
// CHECK-NEXT: template <> struct S<uint64_t &> {};
// CHECK-NEXT: template <> struct S<uint64_t &&> {};
template <> struct S<ulong1> {};
template <> struct S<ulong1 *> {};
template <> struct S<ulong1 &> {};
template <> struct S<ulong1 &&> {};

// CHECK: template <> struct S<sycl::long2> {};
// CHECK-NEXT: template <> struct S<sycl::long2 *> {};
// CHECK-NEXT: template <> struct S<sycl::long2 &> {};
// CHECK-NEXT: template <> struct S<sycl::long2 &&> {};
template <> struct S<long2> {};
template <> struct S<long2 *> {};
template <> struct S<long2 &> {};
template <> struct S<long2 &&> {};

// CHECK: template <> struct S<sycl::ulong2> {};
// CHECK-NEXT: template <> struct S<sycl::ulong2 *> {};
// CHECK-NEXT: template <> struct S<sycl::ulong2 &> {};
// CHECK-NEXT: template <> struct S<sycl::ulong2 &&> {};
template <> struct S<ulong2> {};
template <> struct S<ulong2 *> {};
template <> struct S<ulong2 &> {};
template <> struct S<ulong2 &&> {};

// CHECK: template <> struct S<sycl::long3> {};
// CHECK-NEXT: template <> struct S<sycl::long3 *> {};
// CHECK-NEXT: template <> struct S<sycl::long3 &> {};
// CHECK-NEXT: template <> struct S<sycl::long3 &&> {};
template <> struct S<long3> {};
template <> struct S<long3 *> {};
template <> struct S<long3 &> {};
template <> struct S<long3 &&> {};

// CHECK: template <> struct S<sycl::ulong3> {};
// CHECK-NEXT: template <> struct S<sycl::ulong3 *> {};
// CHECK-NEXT: template <> struct S<sycl::ulong3 &> {};
// CHECK-NEXT: template <> struct S<sycl::ulong3 &&> {};
template <> struct S<ulong3> {};
template <> struct S<ulong3 *> {};
template <> struct S<ulong3 &> {};
template <> struct S<ulong3 &&> {};

// CHECK: template <> struct S<sycl::long4> {};
// CHECK-NEXT: template <> struct S<sycl::long4 *> {};
// CHECK-NEXT: template <> struct S<sycl::long4 &> {};
// CHECK-NEXT: template <> struct S<sycl::long4 &&> {};
template <> struct S<long4> {};
template <> struct S<long4 *> {};
template <> struct S<long4 &> {};
template <> struct S<long4 &&> {};

// CHECK: template <> struct S<sycl::ulong4> {};
// CHECK-NEXT: template <> struct S<sycl::ulong4 *> {};
// CHECK-NEXT: template <> struct S<sycl::ulong4 &> {};
// CHECK-NEXT: template <> struct S<sycl::ulong4 &&> {};
template <> struct S<ulong4> {};
template <> struct S<ulong4 *> {};
template <> struct S<ulong4 &> {};
template <> struct S<ulong4 &&> {};

// CHECK: template <> struct S<float> {};
// CHECK-NEXT: template <> struct S<float *> {};
// CHECK-NEXT: template <> struct S<float &> {};
// CHECK-NEXT: template <> struct S<float &&> {};
template <> struct S<float1> {};
template <> struct S<float1 *> {};
template <> struct S<float1 &> {};
template <> struct S<float1 &&> {};

// CHECK: template <> struct S<sycl::float2> {};
// CHECK-NEXT: template <> struct S<sycl::float2 *> {};
// CHECK-NEXT: template <> struct S<sycl::float2 &> {};
// CHECK-NEXT: template <> struct S<sycl::float2 &&> {};
template <> struct S<float2> {};
template <> struct S<float2 *> {};
template <> struct S<float2 &> {};
template <> struct S<float2 &&> {};

// CHECK: template <> struct S<sycl::float3> {};
// CHECK-NEXT: template <> struct S<sycl::float3 *> {};
// CHECK-NEXT: template <> struct S<sycl::float3 &> {};
// CHECK-NEXT: template <> struct S<sycl::float3 &&> {};
template <> struct S<float3> {};
template <> struct S<float3 *> {};
template <> struct S<float3 &> {};
template <> struct S<float3 &&> {};

// CHECK: template <> struct S<sycl::float4> {};
// CHECK-NEXT: template <> struct S<sycl::float4 *> {};
// CHECK-NEXT: template <> struct S<sycl::float4 &> {};
// CHECK-NEXT: template <> struct S<sycl::float4 &&> {};
template <> struct S<float4> {};
template <> struct S<float4 *> {};
template <> struct S<float4 &> {};
template <> struct S<float4 &&> {};

// CHECK: template <> struct S<int64_t> {};
// CHECK-NEXT: template <> struct S<int64_t *> {};
// CHECK-NEXT: template <> struct S<int64_t &> {};
// CHECK-NEXT: template <> struct S<int64_t &&> {};
template <> struct S<longlong1> {};
template <> struct S<longlong1 *> {};
template <> struct S<longlong1 &> {};
template <> struct S<longlong1 &&> {};

// CHECK: template <> struct S<uint64_t> {};
// CHECK-NEXT: template <> struct S<uint64_t *> {};
// CHECK-NEXT: template <> struct S<uint64_t &> {};
// CHECK-NEXT: template <> struct S<uint64_t &&> {};
template <> struct S<ulonglong1> {};
template <> struct S<ulonglong1 *> {};
template <> struct S<ulonglong1 &> {};
template <> struct S<ulonglong1 &&> {};

// CHECK: template <> struct S<sycl::longlong2> {};
// CHECK-NEXT: template <> struct S<sycl::longlong2 *> {};
// CHECK-NEXT: template <> struct S<sycl::longlong2 &> {};
// CHECK-NEXT: template <> struct S<sycl::longlong2 &&> {};
template <> struct S<longlong2> {};
template <> struct S<longlong2 *> {};
template <> struct S<longlong2 &> {};
template <> struct S<longlong2 &&> {};

// CHECK: template <> struct S<sycl::ulonglong2> {};
// CHECK-NEXT: template <> struct S<sycl::ulonglong2 *> {};
// CHECK-NEXT: template <> struct S<sycl::ulonglong2 &> {};
// CHECK-NEXT: template <> struct S<sycl::ulonglong2 &&> {};
template <> struct S<ulonglong2> {};
template <> struct S<ulonglong2 *> {};
template <> struct S<ulonglong2 &> {};
template <> struct S<ulonglong2 &&> {};

// CHECK: template <> struct S<sycl::longlong3> {};
// CHECK-NEXT: template <> struct S<sycl::longlong3 *> {};
// CHECK-NEXT: template <> struct S<sycl::longlong3 &> {};
// CHECK-NEXT: template <> struct S<sycl::longlong3 &&> {};
template <> struct S<longlong3> {};
template <> struct S<longlong3 *> {};
template <> struct S<longlong3 &> {};
template <> struct S<longlong3 &&> {};

// CHECK: template <> struct S<sycl::ulonglong3> {};
// CHECK-NEXT: template <> struct S<sycl::ulonglong3 *> {};
// CHECK-NEXT: template <> struct S<sycl::ulonglong3 &> {};
// CHECK-NEXT: template <> struct S<sycl::ulonglong3 &&> {};
template <> struct S<ulonglong3> {};
template <> struct S<ulonglong3 *> {};
template <> struct S<ulonglong3 &> {};
template <> struct S<ulonglong3 &&> {};

// CHECK: template <> struct S<sycl::longlong4> {};
// CHECK-NEXT: template <> struct S<sycl::longlong4 *> {};
// CHECK-NEXT: template <> struct S<sycl::longlong4 &> {};
// CHECK-NEXT: template <> struct S<sycl::longlong4 &&> {};
template <> struct S<longlong4> {};
template <> struct S<longlong4 *> {};
template <> struct S<longlong4 &> {};
template <> struct S<longlong4 &&> {};

// CHECK: template <> struct S<sycl::ulonglong4> {};
// CHECK-NEXT: template <> struct S<sycl::ulonglong4 *> {};
// CHECK-NEXT: template <> struct S<sycl::ulonglong4 &> {};
// CHECK-NEXT: template <> struct S<sycl::ulonglong4 &&> {};
template <> struct S<ulonglong4> {};
template <> struct S<ulonglong4 *> {};
template <> struct S<ulonglong4 &> {};
template <> struct S<ulonglong4 &&> {};

// CHECK: template <> struct S<double> {};
// CHECK-NEXT: template <> struct S<double *> {};
// CHECK-NEXT: template <> struct S<double &> {};
// CHECK-NEXT: template <> struct S<double &&> {};
template <> struct S<double1> {};
template <> struct S<double1 *> {};
template <> struct S<double1 &> {};
template <> struct S<double1 &&> {};

// CHECK: template <> struct S<sycl::double2> {};
// CHECK-NEXT: template <> struct S<sycl::double2 *> {};
// CHECK-NEXT: template <> struct S<sycl::double2 &> {};
// CHECK-NEXT: template <> struct S<sycl::double2 &&> {};
template <> struct S<double2> {};
template <> struct S<double2 *> {};
template <> struct S<double2 &> {};
template <> struct S<double2 &&> {};

// CHECK: template <> struct S<sycl::double3> {};
// CHECK-NEXT: template <> struct S<sycl::double3 *> {};
// CHECK-NEXT: template <> struct S<sycl::double3 &> {};
// CHECK-NEXT: template <> struct S<sycl::double3 &&> {};
template <> struct S<double3> {};
template <> struct S<double3 *> {};
template <> struct S<double3 &> {};
template <> struct S<double3 &&> {};

// CHECK: template <> struct S<sycl::double4> {};
// CHECK-NEXT: template <> struct S<sycl::double4 *> {};
// CHECK-NEXT: template <> struct S<sycl::double4 &> {};
// CHECK-NEXT: template <> struct S<sycl::double4 &&> {};
template <> struct S<double4> {};
template <> struct S<double4 *> {};
template <> struct S<double4 &> {};
template <> struct S<double4 &&> {};


// case 5
template <typename T> void template_foo() {}
void case_5(){

// CHECK: template_foo<char>();
// CHECK-NEXT: template_foo<char *>();
// CHECK-NEXT: template_foo<char &>();
// CHECK-NEXT: template_foo<char &&>();
template_foo<char1>();
template_foo<char1 *>();
template_foo<char1 &>();
template_foo<char1 &&>();

// CHECK: template_foo<uint8_t>();
// CHECK-NEXT: template_foo<uint8_t *>();
// CHECK-NEXT: template_foo<uint8_t &>();
// CHECK-NEXT: template_foo<uint8_t &&>();
template_foo<uchar1>();
template_foo<uchar1 *>();
template_foo<uchar1 &>();
template_foo<uchar1 &&>();

// CHECK: template_foo<sycl::char2>();
// CHECK-NEXT: template_foo<sycl::char2 *>();
// CHECK-NEXT: template_foo<sycl::char2 &>();
// CHECK-NEXT: template_foo<sycl::char2 &&>();
template_foo<char2>();
template_foo<char2 *>();
template_foo<char2 &>();
template_foo<char2 &&>();

// CHECK: template_foo<sycl::uchar2>();
// CHECK-NEXT: template_foo<sycl::uchar2 *>();
// CHECK-NEXT: template_foo<sycl::uchar2 &>();
// CHECK-NEXT: template_foo<sycl::uchar2 &&>();
template_foo<uchar2>();
template_foo<uchar2 *>();
template_foo<uchar2 &>();
template_foo<uchar2 &&>();

// CHECK: template_foo<sycl::char3>();
// CHECK-NEXT: template_foo<sycl::char3 *>();
// CHECK-NEXT: template_foo<sycl::char3 &>();
// CHECK-NEXT: template_foo<sycl::char3 &&>();
template_foo<char3>();
template_foo<char3 *>();
template_foo<char3 &>();
template_foo<char3 &&>();

// CHECK: template_foo<sycl::uchar3>();
// CHECK-NEXT: template_foo<sycl::uchar3 *>();
// CHECK-NEXT: template_foo<sycl::uchar3 &>();
// CHECK-NEXT: template_foo<sycl::uchar3 &&>();
template_foo<uchar3>();
template_foo<uchar3 *>();
template_foo<uchar3 &>();
template_foo<uchar3 &&>();

// CHECK: template_foo<sycl::char4>();
// CHECK-NEXT: template_foo<sycl::char4 *>();
// CHECK-NEXT: template_foo<sycl::char4 &>();
// CHECK-NEXT: template_foo<sycl::char4 &&>();
template_foo<char4>();
template_foo<char4 *>();
template_foo<char4 &>();
template_foo<char4 &&>();

// CHECK: template_foo<sycl::uchar4>();
// CHECK-NEXT: template_foo<sycl::uchar4 *>();
// CHECK-NEXT: template_foo<sycl::uchar4 &>();
// CHECK-NEXT: template_foo<sycl::uchar4 &&>();
template_foo<uchar4>();
template_foo<uchar4 *>();
template_foo<uchar4 &>();
template_foo<uchar4 &&>();

// CHECK: template_foo<short>();
// CHECK-NEXT: template_foo<short *>();
// CHECK-NEXT: template_foo<short &>();
// CHECK-NEXT: template_foo<short &&>();
template_foo<short1>();
template_foo<short1 *>();
template_foo<short1 &>();
template_foo<short1 &&>();

// CHECK: template_foo<uint16_t>();
// CHECK-NEXT: template_foo<uint16_t *>();
// CHECK-NEXT: template_foo<uint16_t &>();
// CHECK-NEXT: template_foo<uint16_t &&>();
template_foo<ushort1>();
template_foo<ushort1 *>();
template_foo<ushort1 &>();
template_foo<ushort1 &&>();

// CHECK: template_foo<sycl::short2>();
// CHECK-NEXT: template_foo<sycl::short2 *>();
// CHECK-NEXT: template_foo<sycl::short2 &>();
// CHECK-NEXT: template_foo<sycl::short2 &&>();
template_foo<short2>();
template_foo<short2 *>();
template_foo<short2 &>();
template_foo<short2 &&>();

// CHECK: template_foo<sycl::ushort2>();
// CHECK-NEXT: template_foo<sycl::ushort2 *>();
// CHECK-NEXT: template_foo<sycl::ushort2 &>();
// CHECK-NEXT: template_foo<sycl::ushort2 &&>();
template_foo<ushort2>();
template_foo<ushort2 *>();
template_foo<ushort2 &>();
template_foo<ushort2 &&>();

// CHECK: template_foo<sycl::short3>();
// CHECK-NEXT: template_foo<sycl::short3 *>();
// CHECK-NEXT: template_foo<sycl::short3 &>();
// CHECK-NEXT: template_foo<sycl::short3 &&>();
template_foo<short3>();
template_foo<short3 *>();
template_foo<short3 &>();
template_foo<short3 &&>();

// CHECK: template_foo<sycl::ushort3>();
// CHECK-NEXT: template_foo<sycl::ushort3 *>();
// CHECK-NEXT: template_foo<sycl::ushort3 &>();
// CHECK-NEXT: template_foo<sycl::ushort3 &&>();
template_foo<ushort3>();
template_foo<ushort3 *>();
template_foo<ushort3 &>();
template_foo<ushort3 &&>();

// CHECK: template_foo<sycl::short4>();
// CHECK-NEXT: template_foo<sycl::short4 *>();
// CHECK-NEXT: template_foo<sycl::short4 &>();
// CHECK-NEXT: template_foo<sycl::short4 &&>();
template_foo<short4>();
template_foo<short4 *>();
template_foo<short4 &>();
template_foo<short4 &&>();

// CHECK: template_foo<sycl::ushort4>();
// CHECK-NEXT: template_foo<sycl::ushort4 *>();
// CHECK-NEXT: template_foo<sycl::ushort4 &>();
// CHECK-NEXT: template_foo<sycl::ushort4 &&>();
template_foo<ushort4>();
template_foo<ushort4 *>();
template_foo<ushort4 &>();
template_foo<ushort4 &&>();

// CHECK: template_foo<int>();
// CHECK-NEXT: template_foo<int *>();
// CHECK-NEXT: template_foo<int &>();
// CHECK-NEXT: template_foo<int &&>();
template_foo<int1>();
template_foo<int1 *>();
template_foo<int1 &>();
template_foo<int1 &&>();

// CHECK: template_foo<uint32_t>();
// CHECK-NEXT: template_foo<uint32_t *>();
// CHECK-NEXT: template_foo<uint32_t &>();
// CHECK-NEXT: template_foo<uint32_t &&>();
template_foo<uint1>();
template_foo<uint1 *>();
template_foo<uint1 &>();
template_foo<uint1 &&>();

// CHECK: template_foo<sycl::int2>();
// CHECK-NEXT: template_foo<sycl::int2 *>();
// CHECK-NEXT: template_foo<sycl::int2 &>();
// CHECK-NEXT: template_foo<sycl::int2 &&>();
template_foo<int2>();
template_foo<int2 *>();
template_foo<int2 &>();
template_foo<int2 &&>();

// CHECK: template_foo<sycl::uint2>();
// CHECK-NEXT: template_foo<sycl::uint2 *>();
// CHECK-NEXT: template_foo<sycl::uint2 &>();
// CHECK-NEXT: template_foo<sycl::uint2 &&>();
template_foo<uint2>();
template_foo<uint2 *>();
template_foo<uint2 &>();
template_foo<uint2 &&>();

// CHECK: template_foo<sycl::int3>();
// CHECK-NEXT: template_foo<sycl::int3 *>();
// CHECK-NEXT: template_foo<sycl::int3 &>();
// CHECK-NEXT: template_foo<sycl::int3 &&>();
template_foo<int3>();
template_foo<int3 *>();
template_foo<int3 &>();
template_foo<int3 &&>();

// CHECK: template_foo<sycl::uint3>();
// CHECK-NEXT: template_foo<sycl::uint3 *>();
// CHECK-NEXT: template_foo<sycl::uint3 &>();
// CHECK-NEXT: template_foo<sycl::uint3 &&>();
template_foo<uint3>();
template_foo<uint3 *>();
template_foo<uint3 &>();
template_foo<uint3 &&>();

// CHECK: template_foo<sycl::int4>();
// CHECK-NEXT: template_foo<sycl::int4 *>();
// CHECK-NEXT: template_foo<sycl::int4 &>();
// CHECK-NEXT: template_foo<sycl::int4 &&>();
template_foo<int4>();
template_foo<int4 *>();
template_foo<int4 &>();
template_foo<int4 &&>();

// CHECK: template_foo<sycl::uint4>();
// CHECK-NEXT: template_foo<sycl::uint4 *>();
// CHECK-NEXT: template_foo<sycl::uint4 &>();
// CHECK-NEXT: template_foo<sycl::uint4 &&>();
template_foo<uint4>();
template_foo<uint4 *>();
template_foo<uint4 &>();
template_foo<uint4 &&>();

// CHECK: template_foo<long>();
// CHECK-NEXT: template_foo<long *>();
// CHECK-NEXT: template_foo<long &>();
// CHECK-NEXT: template_foo<long &&>();
template_foo<long1>();
template_foo<long1 *>();
template_foo<long1 &>();
template_foo<long1 &&>();

// CHECK: template_foo<uint64_t>();
// CHECK-NEXT: template_foo<uint64_t *>();
// CHECK-NEXT: template_foo<uint64_t &>();
// CHECK-NEXT: template_foo<uint64_t &&>();
template_foo<ulong1>();
template_foo<ulong1 *>();
template_foo<ulong1 &>();
template_foo<ulong1 &&>();

// CHECK: template_foo<sycl::long2>();
// CHECK-NEXT: template_foo<sycl::long2 *>();
// CHECK-NEXT: template_foo<sycl::long2 &>();
// CHECK-NEXT: template_foo<sycl::long2 &&>();
template_foo<long2>();
template_foo<long2 *>();
template_foo<long2 &>();
template_foo<long2 &&>();

// CHECK: template_foo<sycl::ulong2>();
// CHECK-NEXT: template_foo<sycl::ulong2 *>();
// CHECK-NEXT: template_foo<sycl::ulong2 &>();
// CHECK-NEXT: template_foo<sycl::ulong2 &&>();
template_foo<ulong2>();
template_foo<ulong2 *>();
template_foo<ulong2 &>();
template_foo<ulong2 &&>();

// CHECK: template_foo<sycl::long3>();
// CHECK-NEXT: template_foo<sycl::long3 *>();
// CHECK-NEXT: template_foo<sycl::long3 &>();
// CHECK-NEXT: template_foo<sycl::long3 &&>();
template_foo<long3>();
template_foo<long3 *>();
template_foo<long3 &>();
template_foo<long3 &&>();

// CHECK: template_foo<sycl::ulong3>();
// CHECK-NEXT: template_foo<sycl::ulong3 *>();
// CHECK-NEXT: template_foo<sycl::ulong3 &>();
// CHECK-NEXT: template_foo<sycl::ulong3 &&>();
template_foo<ulong3>();
template_foo<ulong3 *>();
template_foo<ulong3 &>();
template_foo<ulong3 &&>();

// CHECK: template_foo<sycl::long4>();
// CHECK-NEXT: template_foo<sycl::long4 *>();
// CHECK-NEXT: template_foo<sycl::long4 &>();
// CHECK-NEXT: template_foo<sycl::long4 &&>();
template_foo<long4>();
template_foo<long4 *>();
template_foo<long4 &>();
template_foo<long4 &&>();

// CHECK: template_foo<sycl::ulong4>();
// CHECK-NEXT: template_foo<sycl::ulong4 *>();
// CHECK-NEXT: template_foo<sycl::ulong4 &>();
// CHECK-NEXT: template_foo<sycl::ulong4 &&>();
template_foo<ulong4>();
template_foo<ulong4 *>();
template_foo<ulong4 &>();
template_foo<ulong4 &&>();

// CHECK: template_foo<float>();
// CHECK-NEXT: template_foo<float *>();
// CHECK-NEXT: template_foo<float &>();
// CHECK-NEXT: template_foo<float &&>();
template_foo<float1>();
template_foo<float1 *>();
template_foo<float1 &>();
template_foo<float1 &&>();

// CHECK: template_foo<sycl::float2>();
// CHECK-NEXT: template_foo<sycl::float2 *>();
// CHECK-NEXT: template_foo<sycl::float2 &>();
// CHECK-NEXT: template_foo<sycl::float2 &&>();
template_foo<float2>();
template_foo<float2 *>();
template_foo<float2 &>();
template_foo<float2 &&>();

// CHECK: template_foo<sycl::float3>();
// CHECK-NEXT: template_foo<sycl::float3 *>();
// CHECK-NEXT: template_foo<sycl::float3 &>();
// CHECK-NEXT: template_foo<sycl::float3 &&>();
template_foo<float3>();
template_foo<float3 *>();
template_foo<float3 &>();
template_foo<float3 &&>();

// CHECK: template_foo<sycl::float4>();
// CHECK-NEXT: template_foo<sycl::float4 *>();
// CHECK-NEXT: template_foo<sycl::float4 &>();
// CHECK-NEXT: template_foo<sycl::float4 &&>();
template_foo<float4>();
template_foo<float4 *>();
template_foo<float4 &>();
template_foo<float4 &&>();

// CHECK: template_foo<int64_t>();
// CHECK-NEXT: template_foo<int64_t *>();
// CHECK-NEXT: template_foo<int64_t &>();
// CHECK-NEXT: template_foo<int64_t &&>();
template_foo<longlong1>();
template_foo<longlong1 *>();
template_foo<longlong1 &>();
template_foo<longlong1 &&>();

// CHECK: template_foo<uint64_t>();
// CHECK-NEXT: template_foo<uint64_t *>();
// CHECK-NEXT: template_foo<uint64_t &>();
// CHECK-NEXT: template_foo<uint64_t &&>();
template_foo<ulonglong1>();
template_foo<ulonglong1 *>();
template_foo<ulonglong1 &>();
template_foo<ulonglong1 &&>();

// CHECK: template_foo<sycl::longlong2>();
// CHECK-NEXT: template_foo<sycl::longlong2 *>();
// CHECK-NEXT: template_foo<sycl::longlong2 &>();
// CHECK-NEXT: template_foo<sycl::longlong2 &&>();
template_foo<longlong2>();
template_foo<longlong2 *>();
template_foo<longlong2 &>();
template_foo<longlong2 &&>();

// CHECK: template_foo<sycl::ulonglong2>();
// CHECK-NEXT: template_foo<sycl::ulonglong2 *>();
// CHECK-NEXT: template_foo<sycl::ulonglong2 &>();
// CHECK-NEXT: template_foo<sycl::ulonglong2 &&>();
template_foo<ulonglong2>();
template_foo<ulonglong2 *>();
template_foo<ulonglong2 &>();
template_foo<ulonglong2 &&>();

// CHECK: template_foo<sycl::longlong3>();
// CHECK-NEXT: template_foo<sycl::longlong3 *>();
// CHECK-NEXT: template_foo<sycl::longlong3 &>();
// CHECK-NEXT: template_foo<sycl::longlong3 &&>();
template_foo<longlong3>();
template_foo<longlong3 *>();
template_foo<longlong3 &>();
template_foo<longlong3 &&>();

// CHECK: template_foo<sycl::ulonglong3>();
// CHECK-NEXT: template_foo<sycl::ulonglong3 *>();
// CHECK-NEXT: template_foo<sycl::ulonglong3 &>();
// CHECK-NEXT: template_foo<sycl::ulonglong3 &&>();
template_foo<ulonglong3>();
template_foo<ulonglong3 *>();
template_foo<ulonglong3 &>();
template_foo<ulonglong3 &&>();

// CHECK: template_foo<sycl::longlong4>();
// CHECK-NEXT: template_foo<sycl::longlong4 *>();
// CHECK-NEXT: template_foo<sycl::longlong4 &>();
// CHECK-NEXT: template_foo<sycl::longlong4 &&>();
template_foo<longlong4>();
template_foo<longlong4 *>();
template_foo<longlong4 &>();
template_foo<longlong4 &&>();

// CHECK: template_foo<sycl::ulonglong4>();
// CHECK-NEXT: template_foo<sycl::ulonglong4 *>();
// CHECK-NEXT: template_foo<sycl::ulonglong4 &>();
// CHECK-NEXT: template_foo<sycl::ulonglong4 &&>();
template_foo<ulonglong4>();
template_foo<ulonglong4 *>();
template_foo<ulonglong4 &>();
template_foo<ulonglong4 &&>();

// CHECK: template_foo<double>();
// CHECK-NEXT: template_foo<double *>();
// CHECK-NEXT: template_foo<double &>();
// CHECK-NEXT: template_foo<double &&>();
template_foo<double1>();
template_foo<double1 *>();
template_foo<double1 &>();
template_foo<double1 &&>();

// CHECK: template_foo<sycl::double2>();
// CHECK-NEXT: template_foo<sycl::double2 *>();
// CHECK-NEXT: template_foo<sycl::double2 &>();
// CHECK-NEXT: template_foo<sycl::double2 &&>();
template_foo<double2>();
template_foo<double2 *>();
template_foo<double2 &>();
template_foo<double2 &&>();

// CHECK: template_foo<sycl::double3>();
// CHECK-NEXT: template_foo<sycl::double3 *>();
// CHECK-NEXT: template_foo<sycl::double3 &>();
// CHECK-NEXT: template_foo<sycl::double3 &&>();
template_foo<double3>();
template_foo<double3 *>();
template_foo<double3 &>();
template_foo<double3 &&>();

// CHECK: template_foo<sycl::double4>();
// CHECK-NEXT: template_foo<sycl::double4 *>();
// CHECK-NEXT: template_foo<sycl::double4 &>();
// CHECK-NEXT: template_foo<sycl::double4 &&>();
template_foo<double4>();
template_foo<double4 *>();
template_foo<double4 &>();
template_foo<double4 &&>();

}

// case 6

// CHECK: using UT0 = char;
// CHECK-NEXT: using UT1 = char *;
// CHECK-NEXT: using UT2 = char &;
// CHECK-NEXT: using UT3 = char &&;
using UT0 = char1;
using UT1 = char1 *;
using UT2 = char1 &;
using UT3 = char1 &&;

// CHECK: using UT4 = uint8_t;
// CHECK-NEXT: using UT5 = uint8_t *;
// CHECK-NEXT: using UT6 = uint8_t &;
// CHECK-NEXT: using UT7 = uint8_t &&;
using UT4 = uchar1;
using UT5 = uchar1 *;
using UT6 = uchar1 &;
using UT7 = uchar1 &&;

// CHECK: using UT8 = sycl::char2;
// CHECK-NEXT: using UT9 = sycl::char2 *;
// CHECK-NEXT: using UT10 = sycl::char2 &;
// CHECK-NEXT: using UT11 = sycl::char2 &&;
using UT8 = char2;
using UT9 = char2 *;
using UT10 = char2 &;
using UT11 = char2 &&;

// CHECK: using UT12 = sycl::uchar2;
// CHECK-NEXT: using UT13 = sycl::uchar2 *;
// CHECK-NEXT: using UT14 = sycl::uchar2 &;
// CHECK-NEXT: using UT15 = sycl::uchar2 &&;
using UT12 = uchar2;
using UT13 = uchar2 *;
using UT14 = uchar2 &;
using UT15 = uchar2 &&;

// CHECK: using UT16 = sycl::char3;
// CHECK-NEXT: using UT17 = sycl::char3 *;
// CHECK-NEXT: using UT18 = sycl::char3 &;
// CHECK-NEXT: using UT19 = sycl::char3 &&;
using UT16 = char3;
using UT17 = char3 *;
using UT18 = char3 &;
using UT19 = char3 &&;

// CHECK: using UT20 = sycl::uchar3;
// CHECK-NEXT: using UT21 = sycl::uchar3 *;
// CHECK-NEXT: using UT22 = sycl::uchar3 &;
// CHECK-NEXT: using UT23 = sycl::uchar3 &&;
using UT20 = uchar3;
using UT21 = uchar3 *;
using UT22 = uchar3 &;
using UT23 = uchar3 &&;

// CHECK: using UT24 = sycl::char4;
// CHECK-NEXT: using UT25 = sycl::char4 *;
// CHECK-NEXT: using UT26 = sycl::char4 &;
// CHECK-NEXT: using UT27 = sycl::char4 &&;
using UT24 = char4;
using UT25 = char4 *;
using UT26 = char4 &;
using UT27 = char4 &&;

// CHECK: using UT28 = sycl::uchar4;
// CHECK-NEXT: using UT29 = sycl::uchar4 *;
// CHECK-NEXT: using UT30 = sycl::uchar4 &;
// CHECK-NEXT: using UT31 = sycl::uchar4 &&;
using UT28 = uchar4;
using UT29 = uchar4 *;
using UT30 = uchar4 &;
using UT31 = uchar4 &&;

// CHECK: using UT32 = short;
// CHECK-NEXT: using UT33 = short *;
// CHECK-NEXT: using UT34 = short &;
// CHECK-NEXT: using UT35 = short &&;
using UT32 = short1;
using UT33 = short1 *;
using UT34 = short1 &;
using UT35 = short1 &&;

// CHECK: using UT36 = uint16_t;
// CHECK-NEXT: using UT37 = uint16_t *;
// CHECK-NEXT: using UT38 = uint16_t &;
// CHECK-NEXT: using UT39 = uint16_t &&;
using UT36 = ushort1;
using UT37 = ushort1 *;
using UT38 = ushort1 &;
using UT39 = ushort1 &&;

// CHECK: using UT40 = sycl::short2;
// CHECK-NEXT: using UT41 = sycl::short2 *;
// CHECK-NEXT: using UT42 = sycl::short2 &;
// CHECK-NEXT: using UT43 = sycl::short2 &&;
using UT40 = short2;
using UT41 = short2 *;
using UT42 = short2 &;
using UT43 = short2 &&;

// CHECK: using UT44 = sycl::ushort2;
// CHECK-NEXT: using UT45 = sycl::ushort2 *;
// CHECK-NEXT: using UT46 = sycl::ushort2 &;
// CHECK-NEXT: using UT47 = sycl::ushort2 &&;
using UT44 = ushort2;
using UT45 = ushort2 *;
using UT46 = ushort2 &;
using UT47 = ushort2 &&;

// CHECK: using UT48 = sycl::short3;
// CHECK-NEXT: using UT49 = sycl::short3 *;
// CHECK-NEXT: using UT50 = sycl::short3 &;
// CHECK-NEXT: using UT51 = sycl::short3 &&;
using UT48 = short3;
using UT49 = short3 *;
using UT50 = short3 &;
using UT51 = short3 &&;

// CHECK: using UT52 = sycl::ushort3;
// CHECK-NEXT: using UT53 = sycl::ushort3 *;
// CHECK-NEXT: using UT54 = sycl::ushort3 &;
// CHECK-NEXT: using UT55 = sycl::ushort3 &&;
using UT52 = ushort3;
using UT53 = ushort3 *;
using UT54 = ushort3 &;
using UT55 = ushort3 &&;

// CHECK: using UT56 = sycl::short4;
// CHECK-NEXT: using UT57 = sycl::short4 *;
// CHECK-NEXT: using UT58 = sycl::short4 &;
// CHECK-NEXT: using UT59 = sycl::short4 &&;
using UT56 = short4;
using UT57 = short4 *;
using UT58 = short4 &;
using UT59 = short4 &&;

// CHECK: using UT60 = sycl::ushort4;
// CHECK-NEXT: using UT61 = sycl::ushort4 *;
// CHECK-NEXT: using UT62 = sycl::ushort4 &;
// CHECK-NEXT: using UT63 = sycl::ushort4 &&;
using UT60 = ushort4;
using UT61 = ushort4 *;
using UT62 = ushort4 &;
using UT63 = ushort4 &&;

// CHECK: using UT64 = int;
// CHECK-NEXT: using UT65 = int *;
// CHECK-NEXT: using UT66 = int &;
// CHECK-NEXT: using UT67 = int &&;
using UT64 = int1;
using UT65 = int1 *;
using UT66 = int1 &;
using UT67 = int1 &&;

// CHECK: using UT68 = uint32_t;
// CHECK-NEXT: using UT69 = uint32_t *;
// CHECK-NEXT: using UT70 = uint32_t &;
// CHECK-NEXT: using UT71 = uint32_t &&;
using UT68 = uint1;
using UT69 = uint1 *;
using UT70 = uint1 &;
using UT71 = uint1 &&;

// CHECK: using UT72 = sycl::int2;
// CHECK-NEXT: using UT73 = sycl::int2 *;
// CHECK-NEXT: using UT74 = sycl::int2 &;
// CHECK-NEXT: using UT75 = sycl::int2 &&;
using UT72 = int2;
using UT73 = int2 *;
using UT74 = int2 &;
using UT75 = int2 &&;

// CHECK: using UT76 = sycl::uint2;
// CHECK-NEXT: using UT77 = sycl::uint2 *;
// CHECK-NEXT: using UT78 = sycl::uint2 &;
// CHECK-NEXT: using UT79 = sycl::uint2 &&;
using UT76 = uint2;
using UT77 = uint2 *;
using UT78 = uint2 &;
using UT79 = uint2 &&;

// CHECK: using UT80 = sycl::int3;
// CHECK-NEXT: using UT81 = sycl::int3 *;
// CHECK-NEXT: using UT82 = sycl::int3 &;
// CHECK-NEXT: using UT83 = sycl::int3 &&;
using UT80 = int3;
using UT81 = int3 *;
using UT82 = int3 &;
using UT83 = int3 &&;

// CHECK: using UT84 = sycl::uint3;
// CHECK-NEXT: using UT85 = sycl::uint3 *;
// CHECK-NEXT: using UT86 = sycl::uint3 &;
// CHECK-NEXT: using UT87 = sycl::uint3 &&;
using UT84 = uint3;
using UT85 = uint3 *;
using UT86 = uint3 &;
using UT87 = uint3 &&;

// CHECK: using UT88 = sycl::int4;
// CHECK-NEXT: using UT89 = sycl::int4 *;
// CHECK-NEXT: using UT90 = sycl::int4 &;
// CHECK-NEXT: using UT91 = sycl::int4 &&;
using UT88 = int4;
using UT89 = int4 *;
using UT90 = int4 &;
using UT91 = int4 &&;

// CHECK: using UT92 = sycl::uint4;
// CHECK-NEXT: using UT93 = sycl::uint4 *;
// CHECK-NEXT: using UT94 = sycl::uint4 &;
// CHECK-NEXT: using UT95 = sycl::uint4 &&;
using UT92 = uint4;
using UT93 = uint4 *;
using UT94 = uint4 &;
using UT95 = uint4 &&;

// CHECK: using UT96 = long;
// CHECK-NEXT: using UT97 = long *;
// CHECK-NEXT: using UT98 = long &;
// CHECK-NEXT: using UT99 = long &&;
using UT96 = long1;
using UT97 = long1 *;
using UT98 = long1 &;
using UT99 = long1 &&;

// CHECK: using UT100 = uint64_t;
// CHECK-NEXT: using UT101 = uint64_t *;
// CHECK-NEXT: using UT102 = uint64_t &;
// CHECK-NEXT: using UT103 = uint64_t &&;
using UT100 = ulong1;
using UT101 = ulong1 *;
using UT102 = ulong1 &;
using UT103 = ulong1 &&;

// CHECK: using UT104 = sycl::long2;
// CHECK-NEXT: using UT105 = sycl::long2 *;
// CHECK-NEXT: using UT106 = sycl::long2 &;
// CHECK-NEXT: using UT107 = sycl::long2 &&;
using UT104 = long2;
using UT105 = long2 *;
using UT106 = long2 &;
using UT107 = long2 &&;

// CHECK: using UT108 = sycl::ulong2;
// CHECK-NEXT: using UT109 = sycl::ulong2 *;
// CHECK-NEXT: using UT110 = sycl::ulong2 &;
// CHECK-NEXT: using UT111 = sycl::ulong2 &&;
using UT108 = ulong2;
using UT109 = ulong2 *;
using UT110 = ulong2 &;
using UT111 = ulong2 &&;

// CHECK: using UT112 = sycl::long3;
// CHECK-NEXT: using UT113 = sycl::long3 *;
// CHECK-NEXT: using UT114 = sycl::long3 &;
// CHECK-NEXT: using UT115 = sycl::long3 &&;
using UT112 = long3;
using UT113 = long3 *;
using UT114 = long3 &;
using UT115 = long3 &&;

// CHECK: using UT116 = sycl::ulong3;
// CHECK-NEXT: using UT117 = sycl::ulong3 *;
// CHECK-NEXT: using UT118 = sycl::ulong3 &;
// CHECK-NEXT: using UT119 = sycl::ulong3 &&;
using UT116 = ulong3;
using UT117 = ulong3 *;
using UT118 = ulong3 &;
using UT119 = ulong3 &&;

// CHECK: using UT120 = sycl::long4;
// CHECK-NEXT: using UT121 = sycl::long4 *;
// CHECK-NEXT: using UT122 = sycl::long4 &;
// CHECK-NEXT: using UT123 = sycl::long4 &&;
using UT120 = long4;
using UT121 = long4 *;
using UT122 = long4 &;
using UT123 = long4 &&;

// CHECK: using UT124 = sycl::ulong4;
// CHECK-NEXT: using UT125 = sycl::ulong4 *;
// CHECK-NEXT: using UT126 = sycl::ulong4 &;
// CHECK-NEXT: using UT127 = sycl::ulong4 &&;
using UT124 = ulong4;
using UT125 = ulong4 *;
using UT126 = ulong4 &;
using UT127 = ulong4 &&;

// CHECK: using UT128 = float;
// CHECK-NEXT: using UT129 = float *;
// CHECK-NEXT: using UT130 = float &;
// CHECK-NEXT: using UT131 = float &&;
using UT128 = float1;
using UT129 = float1 *;
using UT130 = float1 &;
using UT131 = float1 &&;

// CHECK: using UT132 = sycl::float2;
// CHECK-NEXT: using UT133 = sycl::float2 *;
// CHECK-NEXT: using UT134 = sycl::float2 &;
// CHECK-NEXT: using UT135 = sycl::float2 &&;
using UT132 = float2;
using UT133 = float2 *;
using UT134 = float2 &;
using UT135 = float2 &&;

// CHECK: using UT136 = sycl::float3;
// CHECK-NEXT: using UT137 = sycl::float3 *;
// CHECK-NEXT: using UT138 = sycl::float3 &;
// CHECK-NEXT: using UT139 = sycl::float3 &&;
using UT136 = float3;
using UT137 = float3 *;
using UT138 = float3 &;
using UT139 = float3 &&;

// CHECK: using UT140 = sycl::float4;
// CHECK-NEXT: using UT141 = sycl::float4 *;
// CHECK-NEXT: using UT142 = sycl::float4 &;
// CHECK-NEXT: using UT143 = sycl::float4 &&;
using UT140 = float4;
using UT141 = float4 *;
using UT142 = float4 &;
using UT143 = float4 &&;

// CHECK: using UT144 = int64_t;
// CHECK-NEXT: using UT145 = int64_t *;
// CHECK-NEXT: using UT146 = int64_t &;
// CHECK-NEXT: using UT147 = int64_t &&;
using UT144 = longlong1;
using UT145 = longlong1 *;
using UT146 = longlong1 &;
using UT147 = longlong1 &&;

// CHECK: using UT148 = uint64_t;
// CHECK-NEXT: using UT149 = uint64_t *;
// CHECK-NEXT: using UT150 = uint64_t &;
// CHECK-NEXT: using UT151 = uint64_t &&;
using UT148 = ulonglong1;
using UT149 = ulonglong1 *;
using UT150 = ulonglong1 &;
using UT151 = ulonglong1 &&;

// CHECK: using UT152 = sycl::longlong2;
// CHECK-NEXT: using UT153 = sycl::longlong2 *;
// CHECK-NEXT: using UT154 = sycl::longlong2 &;
// CHECK-NEXT: using UT155 = sycl::longlong2 &&;
using UT152 = longlong2;
using UT153 = longlong2 *;
using UT154 = longlong2 &;
using UT155 = longlong2 &&;

// CHECK: using UT156 = sycl::ulonglong2;
// CHECK-NEXT: using UT157 = sycl::ulonglong2 *;
// CHECK-NEXT: using UT158 = sycl::ulonglong2 &;
// CHECK-NEXT: using UT159 = sycl::ulonglong2 &&;
using UT156 = ulonglong2;
using UT157 = ulonglong2 *;
using UT158 = ulonglong2 &;
using UT159 = ulonglong2 &&;

// CHECK: using UT160 = sycl::longlong3;
// CHECK-NEXT: using UT161 = sycl::longlong3 *;
// CHECK-NEXT: using UT162 = sycl::longlong3 &;
// CHECK-NEXT: using UT163 = sycl::longlong3 &&;
using UT160 = longlong3;
using UT161 = longlong3 *;
using UT162 = longlong3 &;
using UT163 = longlong3 &&;

// CHECK: using UT164 = sycl::ulonglong3;
// CHECK-NEXT: using UT165 = sycl::ulonglong3 *;
// CHECK-NEXT: using UT166 = sycl::ulonglong3 &;
// CHECK-NEXT: using UT167 = sycl::ulonglong3 &&;
using UT164 = ulonglong3;
using UT165 = ulonglong3 *;
using UT166 = ulonglong3 &;
using UT167 = ulonglong3 &&;

// CHECK: using UT168 = sycl::longlong4;
// CHECK-NEXT: using UT169 = sycl::longlong4 *;
// CHECK-NEXT: using UT170 = sycl::longlong4 &;
// CHECK-NEXT: using UT171 = sycl::longlong4 &&;
using UT168 = longlong4;
using UT169 = longlong4 *;
using UT170 = longlong4 &;
using UT171 = longlong4 &&;

// CHECK: using UT172 = sycl::ulonglong4;
// CHECK-NEXT: using UT173 = sycl::ulonglong4 *;
// CHECK-NEXT: using UT174 = sycl::ulonglong4 &;
// CHECK-NEXT: using UT175 = sycl::ulonglong4 &&;
using UT172 = ulonglong4;
using UT173 = ulonglong4 *;
using UT174 = ulonglong4 &;
using UT175 = ulonglong4 &&;

// CHECK: using UT176 = double;
// CHECK-NEXT: using UT177 = double *;
// CHECK-NEXT: using UT178 = double &;
// CHECK-NEXT: using UT179 = double &&;
using UT176 = double1;
using UT177 = double1 *;
using UT178 = double1 &;
using UT179 = double1 &&;

// CHECK: using UT180 = sycl::double2;
// CHECK-NEXT: using UT181 = sycl::double2 *;
// CHECK-NEXT: using UT182 = sycl::double2 &;
// CHECK-NEXT: using UT183 = sycl::double2 &&;
using UT180 = double2;
using UT181 = double2 *;
using UT182 = double2 &;
using UT183 = double2 &&;

// CHECK: using UT184 = sycl::double3;
// CHECK-NEXT: using UT185 = sycl::double3 *;
// CHECK-NEXT: using UT186 = sycl::double3 &;
// CHECK-NEXT: using UT187 = sycl::double3 &&;
using UT184 = double3;
using UT185 = double3 *;
using UT186 = double3 &;
using UT187 = double3 &&;

// CHECK: using UT188 = sycl::double4;
// CHECK-NEXT: using UT189 = sycl::double4 *;
// CHECK-NEXT: using UT190 = sycl::double4 &;
// CHECK-NEXT: using UT191 = sycl::double4 &&;
using UT188 = double4;
using UT189 = double4 *;
using UT190 = double4 &;
using UT191 = double4 &&;


// case 7

// CHECK: typedef char T0;
// CHECK-NEXT: typedef char* T1;
// CHECK-NEXT: typedef char& T2;
// CHECK-NEXT: typedef char&& T3;
typedef char1 T0;
typedef char1* T1;
typedef char1& T2;
typedef char1&& T3;

// CHECK: typedef uint8_t T4;
// CHECK-NEXT: typedef uint8_t* T5;
// CHECK-NEXT: typedef uint8_t& T6;
// CHECK-NEXT: typedef uint8_t&& T7;
typedef uchar1 T4;
typedef uchar1* T5;
typedef uchar1& T6;
typedef uchar1&& T7;

// CHECK: typedef sycl::char2 T8;
// CHECK-NEXT: typedef sycl::char2* T9;
// CHECK-NEXT: typedef sycl::char2& T10;
// CHECK-NEXT: typedef sycl::char2&& T11;
typedef char2 T8;
typedef char2* T9;
typedef char2& T10;
typedef char2&& T11;

// CHECK: typedef sycl::uchar2 T12;
// CHECK-NEXT: typedef sycl::uchar2* T13;
// CHECK-NEXT: typedef sycl::uchar2& T14;
// CHECK-NEXT: typedef sycl::uchar2&& T15;
typedef uchar2 T12;
typedef uchar2* T13;
typedef uchar2& T14;
typedef uchar2&& T15;

// CHECK: typedef sycl::char3 T16;
// CHECK-NEXT: typedef sycl::char3* T17;
// CHECK-NEXT: typedef sycl::char3& T18;
// CHECK-NEXT: typedef sycl::char3&& T19;
typedef char3 T16;
typedef char3* T17;
typedef char3& T18;
typedef char3&& T19;

// CHECK: typedef sycl::uchar3 T20;
// CHECK-NEXT: typedef sycl::uchar3* T21;
// CHECK-NEXT: typedef sycl::uchar3& T22;
// CHECK-NEXT: typedef sycl::uchar3&& T23;
typedef uchar3 T20;
typedef uchar3* T21;
typedef uchar3& T22;
typedef uchar3&& T23;

// CHECK: typedef sycl::char4 T24;
// CHECK-NEXT: typedef sycl::char4* T25;
// CHECK-NEXT: typedef sycl::char4& T26;
// CHECK-NEXT: typedef sycl::char4&& T27;
typedef char4 T24;
typedef char4* T25;
typedef char4& T26;
typedef char4&& T27;

// CHECK: typedef sycl::uchar4 T28;
// CHECK-NEXT: typedef sycl::uchar4* T29;
// CHECK-NEXT: typedef sycl::uchar4& T30;
// CHECK-NEXT: typedef sycl::uchar4&& T31;
typedef uchar4 T28;
typedef uchar4* T29;
typedef uchar4& T30;
typedef uchar4&& T31;

// CHECK: typedef short T32;
// CHECK-NEXT: typedef short* T33;
// CHECK-NEXT: typedef short& T34;
// CHECK-NEXT: typedef short&& T35;
typedef short1 T32;
typedef short1* T33;
typedef short1& T34;
typedef short1&& T35;

// CHECK: typedef uint16_t T36;
// CHECK-NEXT: typedef uint16_t* T37;
// CHECK-NEXT: typedef uint16_t& T38;
// CHECK-NEXT: typedef uint16_t&& T39;
typedef ushort1 T36;
typedef ushort1* T37;
typedef ushort1& T38;
typedef ushort1&& T39;

// CHECK: typedef sycl::short2 T40;
// CHECK-NEXT: typedef sycl::short2* T41;
// CHECK-NEXT: typedef sycl::short2& T42;
// CHECK-NEXT: typedef sycl::short2&& T43;
typedef short2 T40;
typedef short2* T41;
typedef short2& T42;
typedef short2&& T43;

// CHECK: typedef sycl::ushort2 T44;
// CHECK-NEXT: typedef sycl::ushort2* T45;
// CHECK-NEXT: typedef sycl::ushort2& T46;
// CHECK-NEXT: typedef sycl::ushort2&& T47;
typedef ushort2 T44;
typedef ushort2* T45;
typedef ushort2& T46;
typedef ushort2&& T47;

// CHECK: typedef sycl::short3 T48;
// CHECK-NEXT: typedef sycl::short3* T49;
// CHECK-NEXT: typedef sycl::short3& T50;
// CHECK-NEXT: typedef sycl::short3&& T51;
typedef short3 T48;
typedef short3* T49;
typedef short3& T50;
typedef short3&& T51;

// CHECK: typedef sycl::ushort3 T52;
// CHECK-NEXT: typedef sycl::ushort3* T53;
// CHECK-NEXT: typedef sycl::ushort3& T54;
// CHECK-NEXT: typedef sycl::ushort3&& T55;
typedef ushort3 T52;
typedef ushort3* T53;
typedef ushort3& T54;
typedef ushort3&& T55;

// CHECK: typedef sycl::short4 T56;
// CHECK-NEXT: typedef sycl::short4* T57;
// CHECK-NEXT: typedef sycl::short4& T58;
// CHECK-NEXT: typedef sycl::short4&& T59;
typedef short4 T56;
typedef short4* T57;
typedef short4& T58;
typedef short4&& T59;

// CHECK: typedef sycl::ushort4 T60;
// CHECK-NEXT: typedef sycl::ushort4* T61;
// CHECK-NEXT: typedef sycl::ushort4& T62;
// CHECK-NEXT: typedef sycl::ushort4&& T63;
typedef ushort4 T60;
typedef ushort4* T61;
typedef ushort4& T62;
typedef ushort4&& T63;

// CHECK: typedef int T64;
// CHECK-NEXT: typedef int* T65;
// CHECK-NEXT: typedef int& T66;
// CHECK-NEXT: typedef int&& T67;
typedef int1 T64;
typedef int1* T65;
typedef int1& T66;
typedef int1&& T67;

// CHECK: typedef uint32_t T68;
// CHECK-NEXT: typedef uint32_t* T69;
// CHECK-NEXT: typedef uint32_t& T70;
// CHECK-NEXT: typedef uint32_t&& T71;
typedef uint1 T68;
typedef uint1* T69;
typedef uint1& T70;
typedef uint1&& T71;

// CHECK: typedef sycl::int2 T72;
// CHECK-NEXT: typedef sycl::int2* T73;
// CHECK-NEXT: typedef sycl::int2& T74;
// CHECK-NEXT: typedef sycl::int2&& T75;
typedef int2 T72;
typedef int2* T73;
typedef int2& T74;
typedef int2&& T75;

// CHECK: typedef sycl::uint2 T76;
// CHECK-NEXT: typedef sycl::uint2* T77;
// CHECK-NEXT: typedef sycl::uint2& T78;
// CHECK-NEXT: typedef sycl::uint2&& T79;
typedef uint2 T76;
typedef uint2* T77;
typedef uint2& T78;
typedef uint2&& T79;

// CHECK: typedef sycl::int3 T80;
// CHECK-NEXT: typedef sycl::int3* T81;
// CHECK-NEXT: typedef sycl::int3& T82;
// CHECK-NEXT: typedef sycl::int3&& T83;
typedef int3 T80;
typedef int3* T81;
typedef int3& T82;
typedef int3&& T83;

// CHECK: typedef sycl::uint3 T84;
// CHECK-NEXT: typedef sycl::uint3* T85;
// CHECK-NEXT: typedef sycl::uint3& T86;
// CHECK-NEXT: typedef sycl::uint3&& T87;
typedef uint3 T84;
typedef uint3* T85;
typedef uint3& T86;
typedef uint3&& T87;

// CHECK: typedef sycl::int4 T88;
// CHECK-NEXT: typedef sycl::int4* T89;
// CHECK-NEXT: typedef sycl::int4& T90;
// CHECK-NEXT: typedef sycl::int4&& T91;
typedef int4 T88;
typedef int4* T89;
typedef int4& T90;
typedef int4&& T91;

// CHECK: typedef sycl::uint4 T92;
// CHECK-NEXT: typedef sycl::uint4* T93;
// CHECK-NEXT: typedef sycl::uint4& T94;
// CHECK-NEXT: typedef sycl::uint4&& T95;
typedef uint4 T92;
typedef uint4* T93;
typedef uint4& T94;
typedef uint4&& T95;

// CHECK: typedef long T96;
// CHECK-NEXT: typedef long* T97;
// CHECK-NEXT: typedef long& T98;
// CHECK-NEXT: typedef long&& T99;
typedef long1 T96;
typedef long1* T97;
typedef long1& T98;
typedef long1&& T99;

// CHECK: typedef uint64_t T100;
// CHECK-NEXT: typedef uint64_t* T101;
// CHECK-NEXT: typedef uint64_t& T102;
// CHECK-NEXT: typedef uint64_t&& T103;
typedef ulong1 T100;
typedef ulong1* T101;
typedef ulong1& T102;
typedef ulong1&& T103;

// CHECK: typedef sycl::long2 T104;
// CHECK-NEXT: typedef sycl::long2* T105;
// CHECK-NEXT: typedef sycl::long2& T106;
// CHECK-NEXT: typedef sycl::long2&& T107;
typedef long2 T104;
typedef long2* T105;
typedef long2& T106;
typedef long2&& T107;

// CHECK: typedef sycl::ulong2 T108;
// CHECK-NEXT: typedef sycl::ulong2* T109;
// CHECK-NEXT: typedef sycl::ulong2& T110;
// CHECK-NEXT: typedef sycl::ulong2&& T111;
typedef ulong2 T108;
typedef ulong2* T109;
typedef ulong2& T110;
typedef ulong2&& T111;

// CHECK: typedef sycl::long3 T112;
// CHECK-NEXT: typedef sycl::long3* T113;
// CHECK-NEXT: typedef sycl::long3& T114;
// CHECK-NEXT: typedef sycl::long3&& T115;
typedef long3 T112;
typedef long3* T113;
typedef long3& T114;
typedef long3&& T115;

// CHECK: typedef sycl::ulong3 T116;
// CHECK-NEXT: typedef sycl::ulong3* T117;
// CHECK-NEXT: typedef sycl::ulong3& T118;
// CHECK-NEXT: typedef sycl::ulong3&& T119;
typedef ulong3 T116;
typedef ulong3* T117;
typedef ulong3& T118;
typedef ulong3&& T119;

// CHECK: typedef sycl::long4 T120;
// CHECK-NEXT: typedef sycl::long4* T121;
// CHECK-NEXT: typedef sycl::long4& T122;
// CHECK-NEXT: typedef sycl::long4&& T123;
typedef long4 T120;
typedef long4* T121;
typedef long4& T122;
typedef long4&& T123;

// CHECK: typedef sycl::ulong4 T124;
// CHECK-NEXT: typedef sycl::ulong4* T125;
// CHECK-NEXT: typedef sycl::ulong4& T126;
// CHECK-NEXT: typedef sycl::ulong4&& T127;
typedef ulong4 T124;
typedef ulong4* T125;
typedef ulong4& T126;
typedef ulong4&& T127;

// CHECK: typedef float T128;
// CHECK-NEXT: typedef float* T129;
// CHECK-NEXT: typedef float& T130;
// CHECK-NEXT: typedef float&& T131;
typedef float1 T128;
typedef float1* T129;
typedef float1& T130;
typedef float1&& T131;

// CHECK: typedef sycl::float2 T132;
// CHECK-NEXT: typedef sycl::float2* T133;
// CHECK-NEXT: typedef sycl::float2& T134;
// CHECK-NEXT: typedef sycl::float2&& T135;
typedef float2 T132;
typedef float2* T133;
typedef float2& T134;
typedef float2&& T135;

// CHECK: typedef sycl::float3 T136;
// CHECK-NEXT: typedef sycl::float3* T137;
// CHECK-NEXT: typedef sycl::float3& T138;
// CHECK-NEXT: typedef sycl::float3&& T139;
typedef float3 T136;
typedef float3* T137;
typedef float3& T138;
typedef float3&& T139;

// CHECK: typedef sycl::float4 T140;
// CHECK-NEXT: typedef sycl::float4* T141;
// CHECK-NEXT: typedef sycl::float4& T142;
// CHECK-NEXT: typedef sycl::float4&& T143;
typedef float4 T140;
typedef float4* T141;
typedef float4& T142;
typedef float4&& T143;

// CHECK: typedef int64_t T144;
// CHECK-NEXT: typedef int64_t* T145;
// CHECK-NEXT: typedef int64_t& T146;
// CHECK-NEXT: typedef int64_t&& T147;
typedef longlong1 T144;
typedef longlong1* T145;
typedef longlong1& T146;
typedef longlong1&& T147;

// CHECK: typedef uint64_t T148;
// CHECK-NEXT: typedef uint64_t* T149;
// CHECK-NEXT: typedef uint64_t& T150;
// CHECK-NEXT: typedef uint64_t&& T151;
typedef ulonglong1 T148;
typedef ulonglong1* T149;
typedef ulonglong1& T150;
typedef ulonglong1&& T151;

// CHECK: typedef sycl::longlong2 T152;
// CHECK-NEXT: typedef sycl::longlong2* T153;
// CHECK-NEXT: typedef sycl::longlong2& T154;
// CHECK-NEXT: typedef sycl::longlong2&& T155;
typedef longlong2 T152;
typedef longlong2* T153;
typedef longlong2& T154;
typedef longlong2&& T155;

// CHECK: typedef sycl::ulonglong2 T156;
// CHECK-NEXT: typedef sycl::ulonglong2* T157;
// CHECK-NEXT: typedef sycl::ulonglong2& T158;
// CHECK-NEXT: typedef sycl::ulonglong2&& T159;
typedef ulonglong2 T156;
typedef ulonglong2* T157;
typedef ulonglong2& T158;
typedef ulonglong2&& T159;

// CHECK: typedef sycl::longlong3 T160;
// CHECK-NEXT: typedef sycl::longlong3* T161;
// CHECK-NEXT: typedef sycl::longlong3& T162;
// CHECK-NEXT: typedef sycl::longlong3&& T163;
typedef longlong3 T160;
typedef longlong3* T161;
typedef longlong3& T162;
typedef longlong3&& T163;

// CHECK: typedef sycl::ulonglong3 T164;
// CHECK-NEXT: typedef sycl::ulonglong3* T165;
// CHECK-NEXT: typedef sycl::ulonglong3& T166;
// CHECK-NEXT: typedef sycl::ulonglong3&& T167;
typedef ulonglong3 T164;
typedef ulonglong3* T165;
typedef ulonglong3& T166;
typedef ulonglong3&& T167;

// CHECK: typedef sycl::longlong4 T168;
// CHECK-NEXT: typedef sycl::longlong4* T169;
// CHECK-NEXT: typedef sycl::longlong4& T170;
// CHECK-NEXT: typedef sycl::longlong4&& T171;
typedef longlong4 T168;
typedef longlong4* T169;
typedef longlong4& T170;
typedef longlong4&& T171;

// CHECK: typedef sycl::ulonglong4 T172;
// CHECK-NEXT: typedef sycl::ulonglong4* T173;
// CHECK-NEXT: typedef sycl::ulonglong4& T174;
// CHECK-NEXT: typedef sycl::ulonglong4&& T175;
typedef ulonglong4 T172;
typedef ulonglong4* T173;
typedef ulonglong4& T174;
typedef ulonglong4&& T175;

// CHECK: typedef double T176;
// CHECK-NEXT: typedef double* T177;
// CHECK-NEXT: typedef double& T178;
// CHECK-NEXT: typedef double&& T179;
typedef double1 T176;
typedef double1* T177;
typedef double1& T178;
typedef double1&& T179;

// CHECK: typedef sycl::double2 T180;
// CHECK-NEXT: typedef sycl::double2* T181;
// CHECK-NEXT: typedef sycl::double2& T182;
// CHECK-NEXT: typedef sycl::double2&& T183;
typedef double2 T180;
typedef double2* T181;
typedef double2& T182;
typedef double2&& T183;

// CHECK: typedef sycl::double3 T184;
// CHECK-NEXT: typedef sycl::double3* T185;
// CHECK-NEXT: typedef sycl::double3& T186;
// CHECK-NEXT: typedef sycl::double3&& T187;
typedef double3 T184;
typedef double3* T185;
typedef double3& T186;
typedef double3&& T187;

// CHECK: typedef sycl::double4 T188;
// CHECK-NEXT: typedef sycl::double4* T189;
// CHECK-NEXT: typedef sycl::double4& T190;
// CHECK-NEXT: typedef sycl::double4&& T191;
typedef double4 T188;
typedef double4* T189;
typedef double4& T190;
typedef double4&& T191;


// case 8
__device__ void foo_t(){


{
// CHECK: #define T8_0 char
// CHECK-NEXT: #define T8_1 char *
// CHECK-NEXT: #define T8_2 char &
// CHECK-NEXT: #define T8_3 char &&
// CHECK-NEXT:     T8_0 a1;
// CHECK-NEXT:     T8_1 a2;
// CHECK-NEXT:     T8_2 a3=a1;
// CHECK-NEXT:     T8_3 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_0 char1
#define T8_1 char1 *
#define T8_2 char1 &
#define T8_3 char1 &&
    T8_0 a1;
    T8_1 a2;
    T8_2 a3=a1;
    T8_3 a4=std::move(a1);
}

{
// CHECK: #define T8_4 uint8_t
// CHECK-NEXT: #define T8_5 uint8_t *
// CHECK-NEXT: #define T8_6 uint8_t &
// CHECK-NEXT: #define T8_7 uint8_t &&
// CHECK-NEXT:     T8_4 a1;
// CHECK-NEXT:     T8_5 a2;
// CHECK-NEXT:     T8_6 a3=a1;
// CHECK-NEXT:     T8_7 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_4 uchar1
#define T8_5 uchar1 *
#define T8_6 uchar1 &
#define T8_7 uchar1 &&
    T8_4 a1;
    T8_5 a2;
    T8_6 a3=a1;
    T8_7 a4=std::move(a1);
}

{
// CHECK: #define T8_8 sycl::char2
// CHECK-NEXT: #define T8_9 sycl::char2 *
// CHECK-NEXT: #define T8_10 sycl::char2 &
// CHECK-NEXT: #define T8_11 sycl::char2 &&
// CHECK-NEXT:     T8_8 a1;
// CHECK-NEXT:     T8_9 a2;
// CHECK-NEXT:     T8_10 a3=a1;
// CHECK-NEXT:     T8_11 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_8 char2
#define T8_9 char2 *
#define T8_10 char2 &
#define T8_11 char2 &&
    T8_8 a1;
    T8_9 a2;
    T8_10 a3=a1;
    T8_11 a4=std::move(a1);
}

{
// CHECK: #define T8_12 sycl::uchar2
// CHECK-NEXT: #define T8_13 sycl::uchar2 *
// CHECK-NEXT: #define T8_14 sycl::uchar2 &
// CHECK-NEXT: #define T8_15 sycl::uchar2 &&
// CHECK-NEXT:     T8_12 a1;
// CHECK-NEXT:     T8_13 a2;
// CHECK-NEXT:     T8_14 a3=a1;
// CHECK-NEXT:     T8_15 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_12 uchar2
#define T8_13 uchar2 *
#define T8_14 uchar2 &
#define T8_15 uchar2 &&
    T8_12 a1;
    T8_13 a2;
    T8_14 a3=a1;
    T8_15 a4=std::move(a1);
}

{
// CHECK: #define T8_16 sycl::char3
// CHECK-NEXT: #define T8_17 sycl::char3 *
// CHECK-NEXT: #define T8_18 sycl::char3 &
// CHECK-NEXT: #define T8_19 sycl::char3 &&
// CHECK-NEXT:     T8_16 a1;
// CHECK-NEXT:     T8_17 a2;
// CHECK-NEXT:     T8_18 a3=a1;
// CHECK-NEXT:     T8_19 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_16 char3
#define T8_17 char3 *
#define T8_18 char3 &
#define T8_19 char3 &&
    T8_16 a1;
    T8_17 a2;
    T8_18 a3=a1;
    T8_19 a4=std::move(a1);
}

{
// CHECK: #define T8_20 sycl::uchar3
// CHECK-NEXT: #define T8_21 sycl::uchar3 *
// CHECK-NEXT: #define T8_22 sycl::uchar3 &
// CHECK-NEXT: #define T8_23 sycl::uchar3 &&
// CHECK-NEXT:     T8_20 a1;
// CHECK-NEXT:     T8_21 a2;
// CHECK-NEXT:     T8_22 a3=a1;
// CHECK-NEXT:     T8_23 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_20 uchar3
#define T8_21 uchar3 *
#define T8_22 uchar3 &
#define T8_23 uchar3 &&
    T8_20 a1;
    T8_21 a2;
    T8_22 a3=a1;
    T8_23 a4=std::move(a1);
}

{
// CHECK: #define T8_24 sycl::char4
// CHECK-NEXT: #define T8_25 sycl::char4 *
// CHECK-NEXT: #define T8_26 sycl::char4 &
// CHECK-NEXT: #define T8_27 sycl::char4 &&
// CHECK-NEXT:     T8_24 a1;
// CHECK-NEXT:     T8_25 a2;
// CHECK-NEXT:     T8_26 a3=a1;
// CHECK-NEXT:     T8_27 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_24 char4
#define T8_25 char4 *
#define T8_26 char4 &
#define T8_27 char4 &&
    T8_24 a1;
    T8_25 a2;
    T8_26 a3=a1;
    T8_27 a4=std::move(a1);
}

{
// CHECK: #define T8_28 sycl::uchar4
// CHECK-NEXT: #define T8_29 sycl::uchar4 *
// CHECK-NEXT: #define T8_30 sycl::uchar4 &
// CHECK-NEXT: #define T8_31 sycl::uchar4 &&
// CHECK-NEXT:     T8_28 a1;
// CHECK-NEXT:     T8_29 a2;
// CHECK-NEXT:     T8_30 a3=a1;
// CHECK-NEXT:     T8_31 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_28 uchar4
#define T8_29 uchar4 *
#define T8_30 uchar4 &
#define T8_31 uchar4 &&
    T8_28 a1;
    T8_29 a2;
    T8_30 a3=a1;
    T8_31 a4=std::move(a1);
}

{
// CHECK: #define T8_32 short
// CHECK-NEXT: #define T8_33 short *
// CHECK-NEXT: #define T8_34 short &
// CHECK-NEXT: #define T8_35 short &&
// CHECK-NEXT:     T8_32 a1;
// CHECK-NEXT:     T8_33 a2;
// CHECK-NEXT:     T8_34 a3=a1;
// CHECK-NEXT:     T8_35 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_32 short1
#define T8_33 short1 *
#define T8_34 short1 &
#define T8_35 short1 &&
    T8_32 a1;
    T8_33 a2;
    T8_34 a3=a1;
    T8_35 a4=std::move(a1);
}

{
// CHECK: #define T8_36 uint16_t
// CHECK-NEXT: #define T8_37 uint16_t *
// CHECK-NEXT: #define T8_38 uint16_t &
// CHECK-NEXT: #define T8_39 uint16_t &&
// CHECK-NEXT:     T8_36 a1;
// CHECK-NEXT:     T8_37 a2;
// CHECK-NEXT:     T8_38 a3=a1;
// CHECK-NEXT:     T8_39 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_36 ushort1
#define T8_37 ushort1 *
#define T8_38 ushort1 &
#define T8_39 ushort1 &&
    T8_36 a1;
    T8_37 a2;
    T8_38 a3=a1;
    T8_39 a4=std::move(a1);
}

{
// CHECK: #define T8_40 sycl::short2
// CHECK-NEXT: #define T8_41 sycl::short2 *
// CHECK-NEXT: #define T8_42 sycl::short2 &
// CHECK-NEXT: #define T8_43 sycl::short2 &&
// CHECK-NEXT:     T8_40 a1;
// CHECK-NEXT:     T8_41 a2;
// CHECK-NEXT:     T8_42 a3=a1;
// CHECK-NEXT:     T8_43 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_40 short2
#define T8_41 short2 *
#define T8_42 short2 &
#define T8_43 short2 &&
    T8_40 a1;
    T8_41 a2;
    T8_42 a3=a1;
    T8_43 a4=std::move(a1);
}

{
// CHECK: #define T8_44 sycl::ushort2
// CHECK-NEXT: #define T8_45 sycl::ushort2 *
// CHECK-NEXT: #define T8_46 sycl::ushort2 &
// CHECK-NEXT: #define T8_47 sycl::ushort2 &&
// CHECK-NEXT:     T8_44 a1;
// CHECK-NEXT:     T8_45 a2;
// CHECK-NEXT:     T8_46 a3=a1;
// CHECK-NEXT:     T8_47 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_44 ushort2
#define T8_45 ushort2 *
#define T8_46 ushort2 &
#define T8_47 ushort2 &&
    T8_44 a1;
    T8_45 a2;
    T8_46 a3=a1;
    T8_47 a4=std::move(a1);
}

{
// CHECK: #define T8_48 sycl::short3
// CHECK-NEXT: #define T8_49 sycl::short3 *
// CHECK-NEXT: #define T8_50 sycl::short3 &
// CHECK-NEXT: #define T8_51 sycl::short3 &&
// CHECK-NEXT:     T8_48 a1;
// CHECK-NEXT:     T8_49 a2;
// CHECK-NEXT:     T8_50 a3=a1;
// CHECK-NEXT:     T8_51 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_48 short3
#define T8_49 short3 *
#define T8_50 short3 &
#define T8_51 short3 &&
    T8_48 a1;
    T8_49 a2;
    T8_50 a3=a1;
    T8_51 a4=std::move(a1);
}

{
// CHECK: #define T8_52 sycl::ushort3
// CHECK-NEXT: #define T8_53 sycl::ushort3 *
// CHECK-NEXT: #define T8_54 sycl::ushort3 &
// CHECK-NEXT: #define T8_55 sycl::ushort3 &&
// CHECK-NEXT:     T8_52 a1;
// CHECK-NEXT:     T8_53 a2;
// CHECK-NEXT:     T8_54 a3=a1;
// CHECK-NEXT:     T8_55 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_52 ushort3
#define T8_53 ushort3 *
#define T8_54 ushort3 &
#define T8_55 ushort3 &&
    T8_52 a1;
    T8_53 a2;
    T8_54 a3=a1;
    T8_55 a4=std::move(a1);
}

{
// CHECK: #define T8_56 sycl::short4
// CHECK-NEXT: #define T8_57 sycl::short4 *
// CHECK-NEXT: #define T8_58 sycl::short4 &
// CHECK-NEXT: #define T8_59 sycl::short4 &&
// CHECK-NEXT:     T8_56 a1;
// CHECK-NEXT:     T8_57 a2;
// CHECK-NEXT:     T8_58 a3=a1;
// CHECK-NEXT:     T8_59 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_56 short4
#define T8_57 short4 *
#define T8_58 short4 &
#define T8_59 short4 &&
    T8_56 a1;
    T8_57 a2;
    T8_58 a3=a1;
    T8_59 a4=std::move(a1);
}

{
// CHECK: #define T8_60 sycl::ushort4
// CHECK-NEXT: #define T8_61 sycl::ushort4 *
// CHECK-NEXT: #define T8_62 sycl::ushort4 &
// CHECK-NEXT: #define T8_63 sycl::ushort4 &&
// CHECK-NEXT:     T8_60 a1;
// CHECK-NEXT:     T8_61 a2;
// CHECK-NEXT:     T8_62 a3=a1;
// CHECK-NEXT:     T8_63 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_60 ushort4
#define T8_61 ushort4 *
#define T8_62 ushort4 &
#define T8_63 ushort4 &&
    T8_60 a1;
    T8_61 a2;
    T8_62 a3=a1;
    T8_63 a4=std::move(a1);
}

{
// CHECK: #define T8_64 int
// CHECK-NEXT: #define T8_65 int *
// CHECK-NEXT: #define T8_66 int &
// CHECK-NEXT: #define T8_67 int &&
// CHECK-NEXT:     T8_64 a1;
// CHECK-NEXT:     T8_65 a2;
// CHECK-NEXT:     T8_66 a3=a1;
// CHECK-NEXT:     T8_67 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_64 int1
#define T8_65 int1 *
#define T8_66 int1 &
#define T8_67 int1 &&
    T8_64 a1;
    T8_65 a2;
    T8_66 a3=a1;
    T8_67 a4=std::move(a1);
}

{
// CHECK: #define T8_68 uint32_t
// CHECK-NEXT: #define T8_69 uint32_t *
// CHECK-NEXT: #define T8_70 uint32_t &
// CHECK-NEXT: #define T8_71 uint32_t &&
// CHECK-NEXT:     T8_68 a1;
// CHECK-NEXT:     T8_69 a2;
// CHECK-NEXT:     T8_70 a3=a1;
// CHECK-NEXT:     T8_71 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_68 uint1
#define T8_69 uint1 *
#define T8_70 uint1 &
#define T8_71 uint1 &&
    T8_68 a1;
    T8_69 a2;
    T8_70 a3=a1;
    T8_71 a4=std::move(a1);
}

{
// CHECK: #define T8_72 sycl::int2
// CHECK-NEXT: #define T8_73 sycl::int2 *
// CHECK-NEXT: #define T8_74 sycl::int2 &
// CHECK-NEXT: #define T8_75 sycl::int2 &&
// CHECK-NEXT:     T8_72 a1;
// CHECK-NEXT:     T8_73 a2;
// CHECK-NEXT:     T8_74 a3=a1;
// CHECK-NEXT:     T8_75 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_72 int2
#define T8_73 int2 *
#define T8_74 int2 &
#define T8_75 int2 &&
    T8_72 a1;
    T8_73 a2;
    T8_74 a3=a1;
    T8_75 a4=std::move(a1);
}

{
// CHECK: #define T8_76 sycl::uint2
// CHECK-NEXT: #define T8_77 sycl::uint2 *
// CHECK-NEXT: #define T8_78 sycl::uint2 &
// CHECK-NEXT: #define T8_79 sycl::uint2 &&
// CHECK-NEXT:     T8_76 a1;
// CHECK-NEXT:     T8_77 a2;
// CHECK-NEXT:     T8_78 a3=a1;
// CHECK-NEXT:     T8_79 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_76 uint2
#define T8_77 uint2 *
#define T8_78 uint2 &
#define T8_79 uint2 &&
    T8_76 a1;
    T8_77 a2;
    T8_78 a3=a1;
    T8_79 a4=std::move(a1);
}

{
// CHECK: #define T8_80 sycl::int3
// CHECK-NEXT: #define T8_81 sycl::int3 *
// CHECK-NEXT: #define T8_82 sycl::int3 &
// CHECK-NEXT: #define T8_83 sycl::int3 &&
// CHECK-NEXT:     T8_80 a1;
// CHECK-NEXT:     T8_81 a2;
// CHECK-NEXT:     T8_82 a3=a1;
// CHECK-NEXT:     T8_83 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_80 int3
#define T8_81 int3 *
#define T8_82 int3 &
#define T8_83 int3 &&
    T8_80 a1;
    T8_81 a2;
    T8_82 a3=a1;
    T8_83 a4=std::move(a1);
}

{
// CHECK: #define T8_84 sycl::uint3
// CHECK-NEXT: #define T8_85 sycl::uint3 *
// CHECK-NEXT: #define T8_86 sycl::uint3 &
// CHECK-NEXT: #define T8_87 sycl::uint3 &&
// CHECK-NEXT:     T8_84 a1;
// CHECK-NEXT:     T8_85 a2;
// CHECK-NEXT:     T8_86 a3=a1;
// CHECK-NEXT:     T8_87 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_84 uint3
#define T8_85 uint3 *
#define T8_86 uint3 &
#define T8_87 uint3 &&
    T8_84 a1;
    T8_85 a2;
    T8_86 a3=a1;
    T8_87 a4=std::move(a1);
}

{
// CHECK: #define T8_88 sycl::int4
// CHECK-NEXT: #define T8_89 sycl::int4 *
// CHECK-NEXT: #define T8_90 sycl::int4 &
// CHECK-NEXT: #define T8_91 sycl::int4 &&
// CHECK-NEXT:     T8_88 a1;
// CHECK-NEXT:     T8_89 a2;
// CHECK-NEXT:     T8_90 a3=a1;
// CHECK-NEXT:     T8_91 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_88 int4
#define T8_89 int4 *
#define T8_90 int4 &
#define T8_91 int4 &&
    T8_88 a1;
    T8_89 a2;
    T8_90 a3=a1;
    T8_91 a4=std::move(a1);
}

{
// CHECK: #define T8_92 sycl::uint4
// CHECK-NEXT: #define T8_93 sycl::uint4 *
// CHECK-NEXT: #define T8_94 sycl::uint4 &
// CHECK-NEXT: #define T8_95 sycl::uint4 &&
// CHECK-NEXT:     T8_92 a1;
// CHECK-NEXT:     T8_93 a2;
// CHECK-NEXT:     T8_94 a3=a1;
// CHECK-NEXT:     T8_95 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_92 uint4
#define T8_93 uint4 *
#define T8_94 uint4 &
#define T8_95 uint4 &&
    T8_92 a1;
    T8_93 a2;
    T8_94 a3=a1;
    T8_95 a4=std::move(a1);
}

{
// CHECK: #define T8_96 long
// CHECK-NEXT: #define T8_97 long *
// CHECK-NEXT: #define T8_98 long &
// CHECK-NEXT: #define T8_99 long &&
// CHECK-NEXT:     T8_96 a1;
// CHECK-NEXT:     T8_97 a2;
// CHECK-NEXT:     T8_98 a3=a1;
// CHECK-NEXT:     T8_99 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_96 long1
#define T8_97 long1 *
#define T8_98 long1 &
#define T8_99 long1 &&
    T8_96 a1;
    T8_97 a2;
    T8_98 a3=a1;
    T8_99 a4=std::move(a1);
}

{
// CHECK: #define T8_100 uint64_t
// CHECK-NEXT: #define T8_101 uint64_t *
// CHECK-NEXT: #define T8_102 uint64_t &
// CHECK-NEXT: #define T8_103 uint64_t &&
// CHECK-NEXT:     T8_100 a1;
// CHECK-NEXT:     T8_101 a2;
// CHECK-NEXT:     T8_102 a3=a1;
// CHECK-NEXT:     T8_103 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_100 ulong1
#define T8_101 ulong1 *
#define T8_102 ulong1 &
#define T8_103 ulong1 &&
    T8_100 a1;
    T8_101 a2;
    T8_102 a3=a1;
    T8_103 a4=std::move(a1);
}

{
// CHECK: #define T8_104 sycl::long2
// CHECK-NEXT: #define T8_105 sycl::long2 *
// CHECK-NEXT: #define T8_106 sycl::long2 &
// CHECK-NEXT: #define T8_107 sycl::long2 &&
// CHECK-NEXT:     T8_104 a1;
// CHECK-NEXT:     T8_105 a2;
// CHECK-NEXT:     T8_106 a3=a1;
// CHECK-NEXT:     T8_107 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_104 long2
#define T8_105 long2 *
#define T8_106 long2 &
#define T8_107 long2 &&
    T8_104 a1;
    T8_105 a2;
    T8_106 a3=a1;
    T8_107 a4=std::move(a1);
}

{
// CHECK: #define T8_108 sycl::ulong2
// CHECK-NEXT: #define T8_109 sycl::ulong2 *
// CHECK-NEXT: #define T8_110 sycl::ulong2 &
// CHECK-NEXT: #define T8_111 sycl::ulong2 &&
// CHECK-NEXT:     T8_108 a1;
// CHECK-NEXT:     T8_109 a2;
// CHECK-NEXT:     T8_110 a3=a1;
// CHECK-NEXT:     T8_111 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_108 ulong2
#define T8_109 ulong2 *
#define T8_110 ulong2 &
#define T8_111 ulong2 &&
    T8_108 a1;
    T8_109 a2;
    T8_110 a3=a1;
    T8_111 a4=std::move(a1);
}

{
// CHECK: #define T8_112 sycl::long3
// CHECK-NEXT: #define T8_113 sycl::long3 *
// CHECK-NEXT: #define T8_114 sycl::long3 &
// CHECK-NEXT: #define T8_115 sycl::long3 &&
// CHECK-NEXT:     T8_112 a1;
// CHECK-NEXT:     T8_113 a2;
// CHECK-NEXT:     T8_114 a3=a1;
// CHECK-NEXT:     T8_115 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_112 long3
#define T8_113 long3 *
#define T8_114 long3 &
#define T8_115 long3 &&
    T8_112 a1;
    T8_113 a2;
    T8_114 a3=a1;
    T8_115 a4=std::move(a1);
}

{
// CHECK: #define T8_116 sycl::ulong3
// CHECK-NEXT: #define T8_117 sycl::ulong3 *
// CHECK-NEXT: #define T8_118 sycl::ulong3 &
// CHECK-NEXT: #define T8_119 sycl::ulong3 &&
// CHECK-NEXT:     T8_116 a1;
// CHECK-NEXT:     T8_117 a2;
// CHECK-NEXT:     T8_118 a3=a1;
// CHECK-NEXT:     T8_119 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_116 ulong3
#define T8_117 ulong3 *
#define T8_118 ulong3 &
#define T8_119 ulong3 &&
    T8_116 a1;
    T8_117 a2;
    T8_118 a3=a1;
    T8_119 a4=std::move(a1);
}

{
// CHECK: #define T8_120 sycl::long4
// CHECK-NEXT: #define T8_121 sycl::long4 *
// CHECK-NEXT: #define T8_122 sycl::long4 &
// CHECK-NEXT: #define T8_123 sycl::long4 &&
// CHECK-NEXT:     T8_120 a1;
// CHECK-NEXT:     T8_121 a2;
// CHECK-NEXT:     T8_122 a3=a1;
// CHECK-NEXT:     T8_123 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_120 long4
#define T8_121 long4 *
#define T8_122 long4 &
#define T8_123 long4 &&
    T8_120 a1;
    T8_121 a2;
    T8_122 a3=a1;
    T8_123 a4=std::move(a1);
}

{
// CHECK: #define T8_124 sycl::ulong4
// CHECK-NEXT: #define T8_125 sycl::ulong4 *
// CHECK-NEXT: #define T8_126 sycl::ulong4 &
// CHECK-NEXT: #define T8_127 sycl::ulong4 &&
// CHECK-NEXT:     T8_124 a1;
// CHECK-NEXT:     T8_125 a2;
// CHECK-NEXT:     T8_126 a3=a1;
// CHECK-NEXT:     T8_127 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_124 ulong4
#define T8_125 ulong4 *
#define T8_126 ulong4 &
#define T8_127 ulong4 &&
    T8_124 a1;
    T8_125 a2;
    T8_126 a3=a1;
    T8_127 a4=std::move(a1);
}

{
// CHECK: #define T8_128 float
// CHECK-NEXT: #define T8_129 float *
// CHECK-NEXT: #define T8_130 float &
// CHECK-NEXT: #define T8_131 float &&
// CHECK-NEXT:     T8_128 a1;
// CHECK-NEXT:     T8_129 a2;
// CHECK-NEXT:     T8_130 a3=a1;
// CHECK-NEXT:     T8_131 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_128 float1
#define T8_129 float1 *
#define T8_130 float1 &
#define T8_131 float1 &&
    T8_128 a1;
    T8_129 a2;
    T8_130 a3=a1;
    T8_131 a4=std::move(a1);
}

{
// CHECK: #define T8_132 sycl::float2
// CHECK-NEXT: #define T8_133 sycl::float2 *
// CHECK-NEXT: #define T8_134 sycl::float2 &
// CHECK-NEXT: #define T8_135 sycl::float2 &&
// CHECK-NEXT:     T8_132 a1;
// CHECK-NEXT:     T8_133 a2;
// CHECK-NEXT:     T8_134 a3=a1;
// CHECK-NEXT:     T8_135 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_132 float2
#define T8_133 float2 *
#define T8_134 float2 &
#define T8_135 float2 &&
    T8_132 a1;
    T8_133 a2;
    T8_134 a3=a1;
    T8_135 a4=std::move(a1);
}

{
// CHECK: #define T8_136 sycl::float3
// CHECK-NEXT: #define T8_137 sycl::float3 *
// CHECK-NEXT: #define T8_138 sycl::float3 &
// CHECK-NEXT: #define T8_139 sycl::float3 &&
// CHECK-NEXT:     T8_136 a1;
// CHECK-NEXT:     T8_137 a2;
// CHECK-NEXT:     T8_138 a3=a1;
// CHECK-NEXT:     T8_139 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_136 float3
#define T8_137 float3 *
#define T8_138 float3 &
#define T8_139 float3 &&
    T8_136 a1;
    T8_137 a2;
    T8_138 a3=a1;
    T8_139 a4=std::move(a1);
}

{
// CHECK: #define T8_140 sycl::float4
// CHECK-NEXT: #define T8_141 sycl::float4 *
// CHECK-NEXT: #define T8_142 sycl::float4 &
// CHECK-NEXT: #define T8_143 sycl::float4 &&
// CHECK-NEXT:     T8_140 a1;
// CHECK-NEXT:     T8_141 a2;
// CHECK-NEXT:     T8_142 a3=a1;
// CHECK-NEXT:     T8_143 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_140 float4
#define T8_141 float4 *
#define T8_142 float4 &
#define T8_143 float4 &&
    T8_140 a1;
    T8_141 a2;
    T8_142 a3=a1;
    T8_143 a4=std::move(a1);
}

{
// CHECK: #define T8_144 int64_t
// CHECK-NEXT: #define T8_145 int64_t *
// CHECK-NEXT: #define T8_146 int64_t &
// CHECK-NEXT: #define T8_147 int64_t &&
// CHECK-NEXT:     T8_144 a1;
// CHECK-NEXT:     T8_145 a2;
// CHECK-NEXT:     T8_146 a3=a1;
// CHECK-NEXT:     T8_147 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_144 longlong1
#define T8_145 longlong1 *
#define T8_146 longlong1 &
#define T8_147 longlong1 &&
    T8_144 a1;
    T8_145 a2;
    T8_146 a3=a1;
    T8_147 a4=std::move(a1);
}

{
// CHECK: #define T8_148 uint64_t
// CHECK-NEXT: #define T8_149 uint64_t *
// CHECK-NEXT: #define T8_150 uint64_t &
// CHECK-NEXT: #define T8_151 uint64_t &&
// CHECK-NEXT:     T8_148 a1;
// CHECK-NEXT:     T8_149 a2;
// CHECK-NEXT:     T8_150 a3=a1;
// CHECK-NEXT:     T8_151 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_148 ulonglong1
#define T8_149 ulonglong1 *
#define T8_150 ulonglong1 &
#define T8_151 ulonglong1 &&
    T8_148 a1;
    T8_149 a2;
    T8_150 a3=a1;
    T8_151 a4=std::move(a1);
}

{
// CHECK: #define T8_152 sycl::longlong2
// CHECK-NEXT: #define T8_153 sycl::longlong2 *
// CHECK-NEXT: #define T8_154 sycl::longlong2 &
// CHECK-NEXT: #define T8_155 sycl::longlong2 &&
// CHECK-NEXT:     T8_152 a1;
// CHECK-NEXT:     T8_153 a2;
// CHECK-NEXT:     T8_154 a3=a1;
// CHECK-NEXT:     T8_155 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_152 longlong2
#define T8_153 longlong2 *
#define T8_154 longlong2 &
#define T8_155 longlong2 &&
    T8_152 a1;
    T8_153 a2;
    T8_154 a3=a1;
    T8_155 a4=std::move(a1);
}

{
// CHECK: #define T8_156 sycl::ulonglong2
// CHECK-NEXT: #define T8_157 sycl::ulonglong2 *
// CHECK-NEXT: #define T8_158 sycl::ulonglong2 &
// CHECK-NEXT: #define T8_159 sycl::ulonglong2 &&
// CHECK-NEXT:     T8_156 a1;
// CHECK-NEXT:     T8_157 a2;
// CHECK-NEXT:     T8_158 a3=a1;
// CHECK-NEXT:     T8_159 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_156 ulonglong2
#define T8_157 ulonglong2 *
#define T8_158 ulonglong2 &
#define T8_159 ulonglong2 &&
    T8_156 a1;
    T8_157 a2;
    T8_158 a3=a1;
    T8_159 a4=std::move(a1);
}

{
// CHECK: #define T8_160 sycl::longlong3
// CHECK-NEXT: #define T8_161 sycl::longlong3 *
// CHECK-NEXT: #define T8_162 sycl::longlong3 &
// CHECK-NEXT: #define T8_163 sycl::longlong3 &&
// CHECK-NEXT:     T8_160 a1;
// CHECK-NEXT:     T8_161 a2;
// CHECK-NEXT:     T8_162 a3=a1;
// CHECK-NEXT:     T8_163 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_160 longlong3
#define T8_161 longlong3 *
#define T8_162 longlong3 &
#define T8_163 longlong3 &&
    T8_160 a1;
    T8_161 a2;
    T8_162 a3=a1;
    T8_163 a4=std::move(a1);
}

{
// CHECK: #define T8_164 sycl::ulonglong3
// CHECK-NEXT: #define T8_165 sycl::ulonglong3 *
// CHECK-NEXT: #define T8_166 sycl::ulonglong3 &
// CHECK-NEXT: #define T8_167 sycl::ulonglong3 &&
// CHECK-NEXT:     T8_164 a1;
// CHECK-NEXT:     T8_165 a2;
// CHECK-NEXT:     T8_166 a3=a1;
// CHECK-NEXT:     T8_167 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_164 ulonglong3
#define T8_165 ulonglong3 *
#define T8_166 ulonglong3 &
#define T8_167 ulonglong3 &&
    T8_164 a1;
    T8_165 a2;
    T8_166 a3=a1;
    T8_167 a4=std::move(a1);
}

{
// CHECK: #define T8_168 sycl::longlong4
// CHECK-NEXT: #define T8_169 sycl::longlong4 *
// CHECK-NEXT: #define T8_170 sycl::longlong4 &
// CHECK-NEXT: #define T8_171 sycl::longlong4 &&
// CHECK-NEXT:     T8_168 a1;
// CHECK-NEXT:     T8_169 a2;
// CHECK-NEXT:     T8_170 a3=a1;
// CHECK-NEXT:     T8_171 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_168 longlong4
#define T8_169 longlong4 *
#define T8_170 longlong4 &
#define T8_171 longlong4 &&
    T8_168 a1;
    T8_169 a2;
    T8_170 a3=a1;
    T8_171 a4=std::move(a1);
}

{
// CHECK: #define T8_172 sycl::ulonglong4
// CHECK-NEXT: #define T8_173 sycl::ulonglong4 *
// CHECK-NEXT: #define T8_174 sycl::ulonglong4 &
// CHECK-NEXT: #define T8_175 sycl::ulonglong4 &&
// CHECK-NEXT:     T8_172 a1;
// CHECK-NEXT:     T8_173 a2;
// CHECK-NEXT:     T8_174 a3=a1;
// CHECK-NEXT:     T8_175 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_172 ulonglong4
#define T8_173 ulonglong4 *
#define T8_174 ulonglong4 &
#define T8_175 ulonglong4 &&
    T8_172 a1;
    T8_173 a2;
    T8_174 a3=a1;
    T8_175 a4=std::move(a1);
}

{
// CHECK: #define T8_176 double
// CHECK-NEXT: #define T8_177 double *
// CHECK-NEXT: #define T8_178 double &
// CHECK-NEXT: #define T8_179 double &&
// CHECK-NEXT:     T8_176 a1;
// CHECK-NEXT:     T8_177 a2;
// CHECK-NEXT:     T8_178 a3=a1;
// CHECK-NEXT:     T8_179 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_176 double1
#define T8_177 double1 *
#define T8_178 double1 &
#define T8_179 double1 &&
    T8_176 a1;
    T8_177 a2;
    T8_178 a3=a1;
    T8_179 a4=std::move(a1);
}

{
// CHECK: #define T8_180 sycl::double2
// CHECK-NEXT: #define T8_181 sycl::double2 *
// CHECK-NEXT: #define T8_182 sycl::double2 &
// CHECK-NEXT: #define T8_183 sycl::double2 &&
// CHECK-NEXT:     T8_180 a1;
// CHECK-NEXT:     T8_181 a2;
// CHECK-NEXT:     T8_182 a3=a1;
// CHECK-NEXT:     T8_183 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_180 double2
#define T8_181 double2 *
#define T8_182 double2 &
#define T8_183 double2 &&
    T8_180 a1;
    T8_181 a2;
    T8_182 a3=a1;
    T8_183 a4=std::move(a1);
}

{
// CHECK: #define T8_184 sycl::double3
// CHECK-NEXT: #define T8_185 sycl::double3 *
// CHECK-NEXT: #define T8_186 sycl::double3 &
// CHECK-NEXT: #define T8_187 sycl::double3 &&
// CHECK-NEXT:     T8_184 a1;
// CHECK-NEXT:     T8_185 a2;
// CHECK-NEXT:     T8_186 a3=a1;
// CHECK-NEXT:     T8_187 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_184 double3
#define T8_185 double3 *
#define T8_186 double3 &
#define T8_187 double3 &&
    T8_184 a1;
    T8_185 a2;
    T8_186 a3=a1;
    T8_187 a4=std::move(a1);
}

{
// CHECK: #define T8_188 sycl::double4
// CHECK-NEXT: #define T8_189 sycl::double4 *
// CHECK-NEXT: #define T8_190 sycl::double4 &
// CHECK-NEXT: #define T8_191 sycl::double4 &&
// CHECK-NEXT:     T8_188 a1;
// CHECK-NEXT:     T8_189 a2;
// CHECK-NEXT:     T8_190 a3=a1;
// CHECK-NEXT:     T8_191 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_188 double4
#define T8_189 double4 *
#define T8_190 double4 &
#define T8_191 double4 &&
    T8_188 a1;
    T8_189 a2;
    T8_190 a3=a1;
    T8_191 a4=std::move(a1);
}

}

// case 9
template <typename T> void template_foo(T var) {}
#define foo1(DataType) template_foo(DataType varname)
#define foo2(DataType) template_foo(DataType * varname)
#define foo3(DataType) template_foo(DataType & varname)
#define foo4(DataType) template_foo(DataType && varname)

// CHECK: template <> void foo1(char){}
// CHECK-NEXT: template <> void foo2(char){}
// CHECK-NEXT: template <> void foo3(char){}
// CHECK-NEXT: template <> void foo4(char){}
template <> void foo1(char1){}
template <> void foo2(char1){}
template <> void foo3(char1){}
template <> void foo4(char1){}

// CHECK: template <> void foo1(uint8_t){}
// CHECK-NEXT: template <> void foo2(uint8_t){}
// CHECK-NEXT: template <> void foo3(uint8_t){}
// CHECK-NEXT: template <> void foo4(uint8_t){}
template <> void foo1(uchar1){}
template <> void foo2(uchar1){}
template <> void foo3(uchar1){}
template <> void foo4(uchar1){}

// CHECK: template <> void foo1(sycl::char2){}
// CHECK-NEXT: template <> void foo2(sycl::char2){}
// CHECK-NEXT: template <> void foo3(sycl::char2){}
// CHECK-NEXT: template <> void foo4(sycl::char2){}
template <> void foo1(char2){}
template <> void foo2(char2){}
template <> void foo3(char2){}
template <> void foo4(char2){}

// CHECK: template <> void foo1(sycl::uchar2){}
// CHECK-NEXT: template <> void foo2(sycl::uchar2){}
// CHECK-NEXT: template <> void foo3(sycl::uchar2){}
// CHECK-NEXT: template <> void foo4(sycl::uchar2){}
template <> void foo1(uchar2){}
template <> void foo2(uchar2){}
template <> void foo3(uchar2){}
template <> void foo4(uchar2){}

// CHECK: template <> void foo1(sycl::char3){}
// CHECK-NEXT: template <> void foo2(sycl::char3){}
// CHECK-NEXT: template <> void foo3(sycl::char3){}
// CHECK-NEXT: template <> void foo4(sycl::char3){}
template <> void foo1(char3){}
template <> void foo2(char3){}
template <> void foo3(char3){}
template <> void foo4(char3){}

// CHECK: template <> void foo1(sycl::uchar3){}
// CHECK-NEXT: template <> void foo2(sycl::uchar3){}
// CHECK-NEXT: template <> void foo3(sycl::uchar3){}
// CHECK-NEXT: template <> void foo4(sycl::uchar3){}
template <> void foo1(uchar3){}
template <> void foo2(uchar3){}
template <> void foo3(uchar3){}
template <> void foo4(uchar3){}

// CHECK: template <> void foo1(sycl::char4){}
// CHECK-NEXT: template <> void foo2(sycl::char4){}
// CHECK-NEXT: template <> void foo3(sycl::char4){}
// CHECK-NEXT: template <> void foo4(sycl::char4){}
template <> void foo1(char4){}
template <> void foo2(char4){}
template <> void foo3(char4){}
template <> void foo4(char4){}

// CHECK: template <> void foo1(sycl::uchar4){}
// CHECK-NEXT: template <> void foo2(sycl::uchar4){}
// CHECK-NEXT: template <> void foo3(sycl::uchar4){}
// CHECK-NEXT: template <> void foo4(sycl::uchar4){}
template <> void foo1(uchar4){}
template <> void foo2(uchar4){}
template <> void foo3(uchar4){}
template <> void foo4(uchar4){}

// CHECK: template <> void foo1(short){}
// CHECK-NEXT: template <> void foo2(short){}
// CHECK-NEXT: template <> void foo3(short){}
// CHECK-NEXT: template <> void foo4(short){}
template <> void foo1(short1){}
template <> void foo2(short1){}
template <> void foo3(short1){}
template <> void foo4(short1){}

// CHECK: template <> void foo1(uint16_t){}
// CHECK-NEXT: template <> void foo2(uint16_t){}
// CHECK-NEXT: template <> void foo3(uint16_t){}
// CHECK-NEXT: template <> void foo4(uint16_t){}
template <> void foo1(ushort1){}
template <> void foo2(ushort1){}
template <> void foo3(ushort1){}
template <> void foo4(ushort1){}

// CHECK: template <> void foo1(sycl::short2){}
// CHECK-NEXT: template <> void foo2(sycl::short2){}
// CHECK-NEXT: template <> void foo3(sycl::short2){}
// CHECK-NEXT: template <> void foo4(sycl::short2){}
template <> void foo1(short2){}
template <> void foo2(short2){}
template <> void foo3(short2){}
template <> void foo4(short2){}

// CHECK: template <> void foo1(sycl::ushort2){}
// CHECK-NEXT: template <> void foo2(sycl::ushort2){}
// CHECK-NEXT: template <> void foo3(sycl::ushort2){}
// CHECK-NEXT: template <> void foo4(sycl::ushort2){}
template <> void foo1(ushort2){}
template <> void foo2(ushort2){}
template <> void foo3(ushort2){}
template <> void foo4(ushort2){}

// CHECK: template <> void foo1(sycl::short3){}
// CHECK-NEXT: template <> void foo2(sycl::short3){}
// CHECK-NEXT: template <> void foo3(sycl::short3){}
// CHECK-NEXT: template <> void foo4(sycl::short3){}
template <> void foo1(short3){}
template <> void foo2(short3){}
template <> void foo3(short3){}
template <> void foo4(short3){}

// CHECK: template <> void foo1(sycl::ushort3){}
// CHECK-NEXT: template <> void foo2(sycl::ushort3){}
// CHECK-NEXT: template <> void foo3(sycl::ushort3){}
// CHECK-NEXT: template <> void foo4(sycl::ushort3){}
template <> void foo1(ushort3){}
template <> void foo2(ushort3){}
template <> void foo3(ushort3){}
template <> void foo4(ushort3){}

// CHECK: template <> void foo1(sycl::short4){}
// CHECK-NEXT: template <> void foo2(sycl::short4){}
// CHECK-NEXT: template <> void foo3(sycl::short4){}
// CHECK-NEXT: template <> void foo4(sycl::short4){}
template <> void foo1(short4){}
template <> void foo2(short4){}
template <> void foo3(short4){}
template <> void foo4(short4){}

// CHECK: template <> void foo1(sycl::ushort4){}
// CHECK-NEXT: template <> void foo2(sycl::ushort4){}
// CHECK-NEXT: template <> void foo3(sycl::ushort4){}
// CHECK-NEXT: template <> void foo4(sycl::ushort4){}
template <> void foo1(ushort4){}
template <> void foo2(ushort4){}
template <> void foo3(ushort4){}
template <> void foo4(ushort4){}

// CHECK: template <> void foo1(int){}
// CHECK-NEXT: template <> void foo2(int){}
// CHECK-NEXT: template <> void foo3(int){}
// CHECK-NEXT: template <> void foo4(int){}
template <> void foo1(int1){}
template <> void foo2(int1){}
template <> void foo3(int1){}
template <> void foo4(int1){}

// CHECK: template <> void foo1(uint32_t){}
// CHECK-NEXT: template <> void foo2(uint32_t){}
// CHECK-NEXT: template <> void foo3(uint32_t){}
// CHECK-NEXT: template <> void foo4(uint32_t){}
template <> void foo1(uint1){}
template <> void foo2(uint1){}
template <> void foo3(uint1){}
template <> void foo4(uint1){}

// CHECK: template <> void foo1(sycl::int2){}
// CHECK-NEXT: template <> void foo2(sycl::int2){}
// CHECK-NEXT: template <> void foo3(sycl::int2){}
// CHECK-NEXT: template <> void foo4(sycl::int2){}
template <> void foo1(int2){}
template <> void foo2(int2){}
template <> void foo3(int2){}
template <> void foo4(int2){}

// CHECK: template <> void foo1(sycl::uint2){}
// CHECK-NEXT: template <> void foo2(sycl::uint2){}
// CHECK-NEXT: template <> void foo3(sycl::uint2){}
// CHECK-NEXT: template <> void foo4(sycl::uint2){}
template <> void foo1(uint2){}
template <> void foo2(uint2){}
template <> void foo3(uint2){}
template <> void foo4(uint2){}

// CHECK: template <> void foo1(sycl::int3){}
// CHECK-NEXT: template <> void foo2(sycl::int3){}
// CHECK-NEXT: template <> void foo3(sycl::int3){}
// CHECK-NEXT: template <> void foo4(sycl::int3){}
template <> void foo1(int3){}
template <> void foo2(int3){}
template <> void foo3(int3){}
template <> void foo4(int3){}

// CHECK: template <> void foo1(sycl::uint3){}
// CHECK-NEXT: template <> void foo2(sycl::uint3){}
// CHECK-NEXT: template <> void foo3(sycl::uint3){}
// CHECK-NEXT: template <> void foo4(sycl::uint3){}
template <> void foo1(uint3){}
template <> void foo2(uint3){}
template <> void foo3(uint3){}
template <> void foo4(uint3){}

// CHECK: template <> void foo1(sycl::int4){}
// CHECK-NEXT: template <> void foo2(sycl::int4){}
// CHECK-NEXT: template <> void foo3(sycl::int4){}
// CHECK-NEXT: template <> void foo4(sycl::int4){}
template <> void foo1(int4){}
template <> void foo2(int4){}
template <> void foo3(int4){}
template <> void foo4(int4){}

// CHECK: template <> void foo1(sycl::uint4){}
// CHECK-NEXT: template <> void foo2(sycl::uint4){}
// CHECK-NEXT: template <> void foo3(sycl::uint4){}
// CHECK-NEXT: template <> void foo4(sycl::uint4){}
template <> void foo1(uint4){}
template <> void foo2(uint4){}
template <> void foo3(uint4){}
template <> void foo4(uint4){}

// CHECK: template <> void foo1(long){}
// CHECK-NEXT: template <> void foo2(long){}
// CHECK-NEXT: template <> void foo3(long){}
// CHECK-NEXT: template <> void foo4(long){}
template <> void foo1(long1){}
template <> void foo2(long1){}
template <> void foo3(long1){}
template <> void foo4(long1){}

// CHECK: template <> void foo1(uint64_t){}
// CHECK-NEXT: template <> void foo2(uint64_t){}
// CHECK-NEXT: template <> void foo3(uint64_t){}
// CHECK-NEXT: template <> void foo4(uint64_t){}
template <> void foo1(ulong1){}
template <> void foo2(ulong1){}
template <> void foo3(ulong1){}
template <> void foo4(ulong1){}

// CHECK: template <> void foo1(sycl::long2){}
// CHECK-NEXT: template <> void foo2(sycl::long2){}
// CHECK-NEXT: template <> void foo3(sycl::long2){}
// CHECK-NEXT: template <> void foo4(sycl::long2){}
template <> void foo1(long2){}
template <> void foo2(long2){}
template <> void foo3(long2){}
template <> void foo4(long2){}

// CHECK: template <> void foo1(sycl::ulong2){}
// CHECK-NEXT: template <> void foo2(sycl::ulong2){}
// CHECK-NEXT: template <> void foo3(sycl::ulong2){}
// CHECK-NEXT: template <> void foo4(sycl::ulong2){}
template <> void foo1(ulong2){}
template <> void foo2(ulong2){}
template <> void foo3(ulong2){}
template <> void foo4(ulong2){}

// CHECK: template <> void foo1(sycl::long3){}
// CHECK-NEXT: template <> void foo2(sycl::long3){}
// CHECK-NEXT: template <> void foo3(sycl::long3){}
// CHECK-NEXT: template <> void foo4(sycl::long3){}
template <> void foo1(long3){}
template <> void foo2(long3){}
template <> void foo3(long3){}
template <> void foo4(long3){}

// CHECK: template <> void foo1(sycl::ulong3){}
// CHECK-NEXT: template <> void foo2(sycl::ulong3){}
// CHECK-NEXT: template <> void foo3(sycl::ulong3){}
// CHECK-NEXT: template <> void foo4(sycl::ulong3){}
template <> void foo1(ulong3){}
template <> void foo2(ulong3){}
template <> void foo3(ulong3){}
template <> void foo4(ulong3){}

// CHECK: template <> void foo1(sycl::long4){}
// CHECK-NEXT: template <> void foo2(sycl::long4){}
// CHECK-NEXT: template <> void foo3(sycl::long4){}
// CHECK-NEXT: template <> void foo4(sycl::long4){}
template <> void foo1(long4){}
template <> void foo2(long4){}
template <> void foo3(long4){}
template <> void foo4(long4){}

// CHECK: template <> void foo1(sycl::ulong4){}
// CHECK-NEXT: template <> void foo2(sycl::ulong4){}
// CHECK-NEXT: template <> void foo3(sycl::ulong4){}
// CHECK-NEXT: template <> void foo4(sycl::ulong4){}
template <> void foo1(ulong4){}
template <> void foo2(ulong4){}
template <> void foo3(ulong4){}
template <> void foo4(ulong4){}

// CHECK: template <> void foo1(float){}
// CHECK-NEXT: template <> void foo2(float){}
// CHECK-NEXT: template <> void foo3(float){}
// CHECK-NEXT: template <> void foo4(float){}
template <> void foo1(float1){}
template <> void foo2(float1){}
template <> void foo3(float1){}
template <> void foo4(float1){}

// CHECK: template <> void foo1(sycl::float2){}
// CHECK-NEXT: template <> void foo2(sycl::float2){}
// CHECK-NEXT: template <> void foo3(sycl::float2){}
// CHECK-NEXT: template <> void foo4(sycl::float2){}
template <> void foo1(float2){}
template <> void foo2(float2){}
template <> void foo3(float2){}
template <> void foo4(float2){}

// CHECK: template <> void foo1(sycl::float3){}
// CHECK-NEXT: template <> void foo2(sycl::float3){}
// CHECK-NEXT: template <> void foo3(sycl::float3){}
// CHECK-NEXT: template <> void foo4(sycl::float3){}
template <> void foo1(float3){}
template <> void foo2(float3){}
template <> void foo3(float3){}
template <> void foo4(float3){}

// CHECK: template <> void foo1(sycl::float4){}
// CHECK-NEXT: template <> void foo2(sycl::float4){}
// CHECK-NEXT: template <> void foo3(sycl::float4){}
// CHECK-NEXT: template <> void foo4(sycl::float4){}
template <> void foo1(float4){}
template <> void foo2(float4){}
template <> void foo3(float4){}
template <> void foo4(float4){}

// CHECK: template <> void foo1(int64_t){}
// CHECK-NEXT: template <> void foo2(int64_t){}
// CHECK-NEXT: template <> void foo3(int64_t){}
// CHECK-NEXT: template <> void foo4(int64_t){}
template <> void foo1(longlong1){}
template <> void foo2(longlong1){}
template <> void foo3(longlong1){}
template <> void foo4(longlong1){}

// CHECK: template <> void foo1(uint64_t){}
// CHECK-NEXT: template <> void foo2(uint64_t){}
// CHECK-NEXT: template <> void foo3(uint64_t){}
// CHECK-NEXT: template <> void foo4(uint64_t){}
template <> void foo1(ulonglong1){}
template <> void foo2(ulonglong1){}
template <> void foo3(ulonglong1){}
template <> void foo4(ulonglong1){}

// CHECK: template <> void foo1(sycl::longlong2){}
// CHECK-NEXT: template <> void foo2(sycl::longlong2){}
// CHECK-NEXT: template <> void foo3(sycl::longlong2){}
// CHECK-NEXT: template <> void foo4(sycl::longlong2){}
template <> void foo1(longlong2){}
template <> void foo2(longlong2){}
template <> void foo3(longlong2){}
template <> void foo4(longlong2){}

// CHECK: template <> void foo1(sycl::ulonglong2){}
// CHECK-NEXT: template <> void foo2(sycl::ulonglong2){}
// CHECK-NEXT: template <> void foo3(sycl::ulonglong2){}
// CHECK-NEXT: template <> void foo4(sycl::ulonglong2){}
template <> void foo1(ulonglong2){}
template <> void foo2(ulonglong2){}
template <> void foo3(ulonglong2){}
template <> void foo4(ulonglong2){}

// CHECK: template <> void foo1(sycl::longlong3){}
// CHECK-NEXT: template <> void foo2(sycl::longlong3){}
// CHECK-NEXT: template <> void foo3(sycl::longlong3){}
// CHECK-NEXT: template <> void foo4(sycl::longlong3){}
template <> void foo1(longlong3){}
template <> void foo2(longlong3){}
template <> void foo3(longlong3){}
template <> void foo4(longlong3){}

// CHECK: template <> void foo1(sycl::ulonglong3){}
// CHECK-NEXT: template <> void foo2(sycl::ulonglong3){}
// CHECK-NEXT: template <> void foo3(sycl::ulonglong3){}
// CHECK-NEXT: template <> void foo4(sycl::ulonglong3){}
template <> void foo1(ulonglong3){}
template <> void foo2(ulonglong3){}
template <> void foo3(ulonglong3){}
template <> void foo4(ulonglong3){}

// CHECK: template <> void foo1(sycl::longlong4){}
// CHECK-NEXT: template <> void foo2(sycl::longlong4){}
// CHECK-NEXT: template <> void foo3(sycl::longlong4){}
// CHECK-NEXT: template <> void foo4(sycl::longlong4){}
template <> void foo1(longlong4){}
template <> void foo2(longlong4){}
template <> void foo3(longlong4){}
template <> void foo4(longlong4){}

// CHECK: template <> void foo1(sycl::ulonglong4){}
// CHECK-NEXT: template <> void foo2(sycl::ulonglong4){}
// CHECK-NEXT: template <> void foo3(sycl::ulonglong4){}
// CHECK-NEXT: template <> void foo4(sycl::ulonglong4){}
template <> void foo1(ulonglong4){}
template <> void foo2(ulonglong4){}
template <> void foo3(ulonglong4){}
template <> void foo4(ulonglong4){}

// CHECK: template <> void foo1(double){}
// CHECK-NEXT: template <> void foo2(double){}
// CHECK-NEXT: template <> void foo3(double){}
// CHECK-NEXT: template <> void foo4(double){}
template <> void foo1(double1){}
template <> void foo2(double1){}
template <> void foo3(double1){}
template <> void foo4(double1){}

// CHECK: template <> void foo1(sycl::double2){}
// CHECK-NEXT: template <> void foo2(sycl::double2){}
// CHECK-NEXT: template <> void foo3(sycl::double2){}
// CHECK-NEXT: template <> void foo4(sycl::double2){}
template <> void foo1(double2){}
template <> void foo2(double2){}
template <> void foo3(double2){}
template <> void foo4(double2){}

// CHECK: template <> void foo1(sycl::double3){}
// CHECK-NEXT: template <> void foo2(sycl::double3){}
// CHECK-NEXT: template <> void foo3(sycl::double3){}
// CHECK-NEXT: template <> void foo4(sycl::double3){}
template <> void foo1(double3){}
template <> void foo2(double3){}
template <> void foo3(double3){}
template <> void foo4(double3){}

// CHECK: template <> void foo1(sycl::double4){}
// CHECK-NEXT: template <> void foo2(sycl::double4){}
// CHECK-NEXT: template <> void foo3(sycl::double4){}
// CHECK-NEXT: template <> void foo4(sycl::double4){}
template <> void foo1(double4){}
template <> void foo2(double4){}
template <> void foo3(double4){}
template <> void foo4(double4){}

// CHECK: template <> struct S<int> {};
// CHECK-NEXT: template <> struct S<int *> {};
// CHECK-NEXT: template <> struct S<int &> {};
// CHECK-NEXT: template <> struct S<int &&> {};
template <> struct S<cublasStatus_t> {};
template <> struct S<cublasStatus_t *> {};
template <> struct S<cublasStatus_t &> {};
template <> struct S<cublasStatus_t &&> {};

// CHECK: template <> struct S<sycl::queue> {};
// CHECK-NEXT: template <> struct S<sycl::queue *> {};
// CHECK-NEXT: template <> struct S<sycl::queue &> {};
// CHECK-NEXT: template <> struct S<sycl::queue &&> {};
template <> struct S<CUstream_st> {};
template <> struct S<CUstream_st *> {};
template <> struct S<CUstream_st &> {};
template <> struct S<CUstream_st &&> {};

// CHECK: template <> void foo2(sycl::queue){}
// CHECK-NEXT: template <> void foo3(sycl::queue){}
// CHECK-NEXT: template <> void foo4(sycl::queue){}
template <> void foo2(CUstream_st){}
template <> void foo3(CUstream_st){}
template <> void foo4(CUstream_st){}

