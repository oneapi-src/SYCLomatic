// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none --usm-level=none -out-root %T/vector_type_cub %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/vector_type_cub/vector_type_cub.dp.cpp --match-full-lines %s

#include <cub/cub.cuh>

__device__ char1 operator+(char1 a, char1 b) {
  // CHECK: return char(a + b);
  return make_char1(a.x + b.x);
}

__global__ void test_make_char1() {
  typedef cub::BlockReduce<char1, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: char res = sycl::reduce_over_group({{.+}}, char(1.), {{.+}});
  char1 res = BlockReduce(smem_storage).Sum(make_char1(1.));
}

__device__ char2 operator+(char2 a, char2 b) {
  // CHECK: return sycl::mchar2(a[0] + b[0], a[1] + b[1]);
  return make_char2(a.x + b.x, a.y + b.y);
}

__global__ void test_make_char2() {
  typedef cub::BlockReduce<char2, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: sycl::mchar2 res = sycl::reduce_over_group({{.+}}, sycl::mchar2(1., 2.), {{.+}});
  char2 res = BlockReduce(smem_storage).Sum(make_char2(1., 2.));
}

__device__ char3 operator+(char3 a, char3 b) {
  // CHECK: return sycl::mchar3(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
  return make_char3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__global__ void test_make_char3() {
  typedef cub::BlockReduce<char3, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: sycl::mchar3 res = sycl::reduce_over_group({{.+}}, sycl::mchar3(1., 2., 3.), {{.+}});
  char3 res = BlockReduce(smem_storage).Sum(make_char3(1., 2., 3.));
}

__device__ char4 operator+(char4 a, char4 b) {
  // CHECK: return sycl::mchar4(a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]);
  return make_char4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__global__ void test_make_char4() {
  typedef cub::BlockReduce<char4, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: sycl::mchar4 res = sycl::reduce_over_group({{.+}}, sycl::mchar4(1., 2., 3., 4.), {{.+}});
  char4 res = BlockReduce(smem_storage).Sum(make_char4(1., 2., 3., 4.));
}

__device__ uchar1 operator+(uchar1 a, uchar1 b) {
  // CHECK: return uint8_t(a + b);
  return make_uchar1(a.x + b.x);
}

__global__ void test_make_uchar1() {
  typedef cub::BlockReduce<uchar1, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: uint8_t res = sycl::reduce_over_group({{.+}}, uint8_t(1.), {{.+}});
  uchar1 res = BlockReduce(smem_storage).Sum(make_uchar1(1.));
}

__device__ uchar2 operator+(uchar2 a, uchar2 b) {
  // CHECK: return sycl::muchar2(a[0] + b[0], a[1] + b[1]);
  return make_uchar2(a.x + b.x, a.y + b.y);
}

__global__ void test_make_uchar2() {
  typedef cub::BlockReduce<uchar2, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: sycl::muchar2 res = sycl::reduce_over_group({{.+}}, sycl::muchar2(1., 2.), {{.+}});
  uchar2 res = BlockReduce(smem_storage).Sum(make_uchar2(1., 2.));
}

__device__ uchar3 operator+(uchar3 a, uchar3 b) {
  // CHECK: return sycl::muchar3(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
  return make_uchar3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__global__ void test_make_uchar3() {
  typedef cub::BlockReduce<uchar3, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: sycl::muchar3 res = sycl::reduce_over_group({{.+}}, sycl::muchar3(1., 2., 3.), {{.+}});
  uchar3 res = BlockReduce(smem_storage).Sum(make_uchar3(1., 2., 3.));
}

__device__ uchar4 operator+(uchar4 a, uchar4 b) {
  // CHECK: return sycl::muchar4(a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]);
  return make_uchar4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__global__ void test_make_uchar4() {
  typedef cub::BlockReduce<uchar4, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: sycl::muchar4 res = sycl::reduce_over_group({{.+}}, sycl::muchar4(1., 2., 3., 4.), {{.+}});
  uchar4 res = BlockReduce(smem_storage).Sum(make_uchar4(1., 2., 3., 4.));
}

__device__ short1 operator+(short1 a, short1 b) {
  // CHECK: return short(a + b);
  return make_short1(a.x + b.x);
}

__global__ void test_make_short1() {
  typedef cub::BlockReduce<short1, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: short res = sycl::reduce_over_group({{.+}}, short(1.), {{.+}});
  short1 res = BlockReduce(smem_storage).Sum(make_short1(1.));
}

__device__ short2 operator+(short2 a, short2 b) {
  // CHECK: return sycl::mshort2(a[0] + b[0], a[1] + b[1]);
  return make_short2(a.x + b.x, a.y + b.y);
}

__global__ void test_make_short2() {
  typedef cub::BlockReduce<short2, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: sycl::mshort2 res = sycl::reduce_over_group({{.+}}, sycl::mshort2(1., 2.), {{.+}});
  short2 res = BlockReduce(smem_storage).Sum(make_short2(1., 2.));
}

__device__ short3 operator+(short3 a, short3 b) {
  // CHECK: return sycl::mshort3(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
  return make_short3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__global__ void test_make_short3() {
  typedef cub::BlockReduce<short3, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: sycl::mshort3 res = sycl::reduce_over_group({{.+}}, sycl::mshort3(1., 2., 3.), {{.+}});
  short3 res = BlockReduce(smem_storage).Sum(make_short3(1., 2., 3.));
}

__device__ short4 operator+(short4 a, short4 b) {
  // CHECK: return sycl::mshort4(a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]);
  return make_short4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__global__ void test_make_short4() {
  typedef cub::BlockReduce<short4, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: sycl::mshort4 res = sycl::reduce_over_group({{.+}}, sycl::mshort4(1., 2., 3., 4.), {{.+}});
  short4 res = BlockReduce(smem_storage).Sum(make_short4(1., 2., 3., 4.));
}

__device__ ushort1 operator+(ushort1 a, ushort1 b) {
  // CHECK: return uint16_t(a + b);
  return make_ushort1(a.x + b.x);
}

__global__ void test_make_ushort1() {
  typedef cub::BlockReduce<ushort1, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: uint16_t res = sycl::reduce_over_group({{.+}}, uint16_t(1.), {{.+}});
  ushort1 res = BlockReduce(smem_storage).Sum(make_ushort1(1.));
}

__device__ ushort2 operator+(ushort2 a, ushort2 b) {
  // CHECK: return sycl::mushort2(a[0] + b[0], a[1] + b[1]);
  return make_ushort2(a.x + b.x, a.y + b.y);
}

__global__ void test_make_ushort2() {
  typedef cub::BlockReduce<ushort2, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: sycl::mushort2 res = sycl::reduce_over_group({{.+}}, sycl::mushort2(1., 2.), {{.+}});
  ushort2 res = BlockReduce(smem_storage).Sum(make_ushort2(1., 2.));
}

__device__ ushort3 operator+(ushort3 a, ushort3 b) {
  // CHECK: return sycl::mushort3(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
  return make_ushort3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__global__ void test_make_ushort3() {
  typedef cub::BlockReduce<ushort3, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: sycl::mushort3 res = sycl::reduce_over_group({{.+}}, sycl::mushort3(1., 2., 3.), {{.+}});
  ushort3 res = BlockReduce(smem_storage).Sum(make_ushort3(1., 2., 3.));
}

__device__ ushort4 operator+(ushort4 a, ushort4 b) {
  // CHECK: return sycl::mushort4(a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]);
  return make_ushort4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__global__ void test_make_ushort4() {
  typedef cub::BlockReduce<ushort4, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: sycl::mushort4 res = sycl::reduce_over_group({{.+}}, sycl::mushort4(1., 2., 3., 4.), {{.+}});
  ushort4 res = BlockReduce(smem_storage).Sum(make_ushort4(1., 2., 3., 4.));
}

__device__ int1 operator+(int1 a, int1 b) {
  // CHECK: return int(a + b);
  return make_int1(a.x + b.x);
}

__global__ void test_make_int1() {
  typedef cub::BlockReduce<int1, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: int res = sycl::reduce_over_group({{.+}}, int(1.), {{.+}});
  int1 res = BlockReduce(smem_storage).Sum(make_int1(1.));
}

__device__ int2 operator+(int2 a, int2 b) {
  // CHECK: return sycl::mint2(a[0] + b[0], a[1] + b[1]);
  return make_int2(a.x + b.x, a.y + b.y);
}

__global__ void test_make_int2() {
  typedef cub::BlockReduce<int2, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: sycl::mint2 res = sycl::reduce_over_group({{.+}}, sycl::mint2(1., 2.), {{.+}});
  int2 res = BlockReduce(smem_storage).Sum(make_int2(1., 2.));
}

__device__ int3 operator+(int3 a, int3 b) {
  // CHECK: return sycl::mint3(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
  return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__global__ void test_make_int3() {
  typedef cub::BlockReduce<int3, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: sycl::mint3 res = sycl::reduce_over_group({{.+}}, sycl::mint3(1., 2., 3.), {{.+}});
  int3 res = BlockReduce(smem_storage).Sum(make_int3(1., 2., 3.));
}

__device__ int4 operator+(int4 a, int4 b) {
  // CHECK: return sycl::mint4(a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]);
  return make_int4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__global__ void test_make_int4() {
  typedef cub::BlockReduce<int4, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: sycl::mint4 res = sycl::reduce_over_group({{.+}}, sycl::mint4(1., 2., 3., 4.), {{.+}});
  int4 res = BlockReduce(smem_storage).Sum(make_int4(1., 2., 3., 4.));
}

__device__ uint1 operator+(uint1 a, uint1 b) {
  // CHECK: return uint32_t(a + b);
  return make_uint1(a.x + b.x);
}

__global__ void test_make_uint1() {
  typedef cub::BlockReduce<uint1, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: uint32_t res = sycl::reduce_over_group({{.+}}, uint32_t(1.), {{.+}});
  uint1 res = BlockReduce(smem_storage).Sum(make_uint1(1.));
}

__device__ uint2 operator+(uint2 a, uint2 b) {
  // CHECK: return sycl::muint2(a[0] + b[0], a[1] + b[1]);
  return make_uint2(a.x + b.x, a.y + b.y);
}

__global__ void test_make_uint2() {
  typedef cub::BlockReduce<uint2, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: sycl::muint2 res = sycl::reduce_over_group({{.+}}, sycl::muint2(1., 2.), {{.+}});
  uint2 res = BlockReduce(smem_storage).Sum(make_uint2(1., 2.));
}

__device__ uint3 operator+(uint3 a, uint3 b) {
  // CHECK: return sycl::muint3(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
  return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__global__ void test_make_uint3() {
  typedef cub::BlockReduce<uint3, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: sycl::muint3 res = sycl::reduce_over_group({{.+}}, sycl::muint3(1., 2., 3.), {{.+}});
  uint3 res = BlockReduce(smem_storage).Sum(make_uint3(1., 2., 3.));
}

__device__ uint4 operator+(uint4 a, uint4 b) {
  // CHECK: return sycl::muint4(a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]);
  return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__global__ void test_make_uint4() {
  typedef cub::BlockReduce<uint4, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: sycl::muint4 res = sycl::reduce_over_group({{.+}}, sycl::muint4(1., 2., 3., 4.), {{.+}});
  uint4 res = BlockReduce(smem_storage).Sum(make_uint4(1., 2., 3., 4.));
}

__device__ long1 operator+(long1 a, long1 b) {
  // CHECK: return long(a + b);
  return make_long1(a.x + b.x);
}

__global__ void test_make_long1() {
  typedef cub::BlockReduce<long1, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: long res = sycl::reduce_over_group({{.+}}, long(1.), {{.+}});
  long1 res = BlockReduce(smem_storage).Sum(make_long1(1.));
}

__device__ long2 operator+(long2 a, long2 b) {
  // CHECK: return sycl::mlong2(a[0] + b[0], a[1] + b[1]);
  return make_long2(a.x + b.x, a.y + b.y);
}

__global__ void test_make_long2() {
  typedef cub::BlockReduce<long2, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: sycl::mlong2 res = sycl::reduce_over_group({{.+}}, sycl::mlong2(1., 2.), {{.+}});
  long2 res = BlockReduce(smem_storage).Sum(make_long2(1., 2.));
}

__device__ long3 operator+(long3 a, long3 b) {
  // CHECK: return sycl::mlong3(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
  return make_long3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__global__ void test_make_long3() {
  typedef cub::BlockReduce<long3, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: sycl::mlong3 res = sycl::reduce_over_group({{.+}}, sycl::mlong3(1., 2., 3.), {{.+}});
  long3 res = BlockReduce(smem_storage).Sum(make_long3(1., 2., 3.));
}

__device__ long4 operator+(long4 a, long4 b) {
  // CHECK: return sycl::mlong4(a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]);
  return make_long4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__global__ void test_make_long4() {
  typedef cub::BlockReduce<long4, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: sycl::mlong4 res = sycl::reduce_over_group({{.+}}, sycl::mlong4(1., 2., 3., 4.), {{.+}});
  long4 res = BlockReduce(smem_storage).Sum(make_long4(1., 2., 3., 4.));
}

__device__ ulong1 operator+(ulong1 a, ulong1 b) {
  // CHECK: return uint64_t(a + b);
  return make_ulong1(a.x + b.x);
}

__global__ void test_make_ulong1() {
  typedef cub::BlockReduce<ulong1, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: uint64_t res = sycl::reduce_over_group({{.+}}, uint64_t(1.), {{.+}});
  ulong1 res = BlockReduce(smem_storage).Sum(make_ulong1(1.));
}

__device__ ulong2 operator+(ulong2 a, ulong2 b) {
  // CHECK: return sycl::mulong2(a[0] + b[0], a[1] + b[1]);
  return make_ulong2(a.x + b.x, a.y + b.y);
}

__global__ void test_make_ulong2() {
  typedef cub::BlockReduce<ulong2, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: sycl::mulong2 res = sycl::reduce_over_group({{.+}}, sycl::mulong2(1., 2.), {{.+}});
  ulong2 res = BlockReduce(smem_storage).Sum(make_ulong2(1., 2.));
}

__device__ ulong3 operator+(ulong3 a, ulong3 b) {
  // CHECK: return sycl::mulong3(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
  return make_ulong3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__global__ void test_make_ulong3() {
  typedef cub::BlockReduce<ulong3, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: sycl::mulong3 res = sycl::reduce_over_group({{.+}}, sycl::mulong3(1., 2., 3.), {{.+}});
  ulong3 res = BlockReduce(smem_storage).Sum(make_ulong3(1., 2., 3.));
}

__device__ ulong4 operator+(ulong4 a, ulong4 b) {
  // CHECK: return sycl::mulong4(a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]);
  return make_ulong4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__global__ void test_make_ulong4() {
  typedef cub::BlockReduce<ulong4, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: sycl::mulong4 res = sycl::reduce_over_group({{.+}}, sycl::mulong4(1., 2., 3., 4.), {{.+}});
  ulong4 res = BlockReduce(smem_storage).Sum(make_ulong4(1., 2., 3., 4.));
}

__device__ longlong1 operator+(longlong1 a, longlong1 b) {
  // CHECK: return int64_t(a + b);
  return make_longlong1(a.x + b.x);
}

__global__ void test_make_longlong1() {
  typedef cub::BlockReduce<longlong1, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: int64_t res = sycl::reduce_over_group({{.+}}, int64_t(1.), {{.+}});
  longlong1 res = BlockReduce(smem_storage).Sum(make_longlong1(1.));
}

__device__ longlong2 operator+(longlong2 a, longlong2 b) {
  // CHECK: return sycl::mlonglong2(a[0] + b[0], a[1] + b[1]);
  return make_longlong2(a.x + b.x, a.y + b.y);
}

__global__ void test_make_longlong2() {
  typedef cub::BlockReduce<longlong2, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: sycl::mlonglong2 res = sycl::reduce_over_group({{.+}}, sycl::mlonglong2(1., 2.), {{.+}});
  longlong2 res = BlockReduce(smem_storage).Sum(make_longlong2(1., 2.));
}

__device__ longlong3 operator+(longlong3 a, longlong3 b) {
  // CHECK: return sycl::mlonglong3(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
  return make_longlong3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__global__ void test_make_longlong3() {
  typedef cub::BlockReduce<longlong3, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: sycl::mlonglong3 res = sycl::reduce_over_group({{.+}}, sycl::mlonglong3(1., 2., 3.), {{.+}});
  longlong3 res = BlockReduce(smem_storage).Sum(make_longlong3(1., 2., 3.));
}

__device__ longlong4 operator+(longlong4 a, longlong4 b) {
  // CHECK: return sycl::mlonglong4(a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]);
  return make_longlong4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__global__ void test_make_longlong4() {
  typedef cub::BlockReduce<longlong4, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: sycl::mlonglong4 res = sycl::reduce_over_group({{.+}}, sycl::mlonglong4(1., 2., 3., 4.), {{.+}});
  longlong4 res = BlockReduce(smem_storage).Sum(make_longlong4(1., 2., 3., 4.));
}

__device__ ulonglong1 operator+(ulonglong1 a, ulonglong1 b) {
  // CHECK: return uint64_t(a + b);
  return make_ulonglong1(a.x + b.x);
}

__global__ void test_make_ulonglong1() {
  typedef cub::BlockReduce<ulonglong1, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: uint64_t res = sycl::reduce_over_group({{.+}}, uint64_t(1.), {{.+}});
  ulonglong1 res = BlockReduce(smem_storage).Sum(make_ulonglong1(1.));
}

__device__ ulonglong2 operator+(ulonglong2 a, ulonglong2 b) {
  // CHECK: return sycl::mulonglong2(a[0] + b[0], a[1] + b[1]);
  return make_ulonglong2(a.x + b.x, a.y + b.y);
}

__global__ void test_make_ulonglong2() {
  typedef cub::BlockReduce<ulonglong2, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: sycl::mulonglong2 res = sycl::reduce_over_group({{.+}}, sycl::mulonglong2(1., 2.), {{.+}});
  ulonglong2 res = BlockReduce(smem_storage).Sum(make_ulonglong2(1., 2.));
}

__device__ ulonglong3 operator+(ulonglong3 a, ulonglong3 b) {
  // CHECK: return sycl::mulonglong3(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
  return make_ulonglong3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__global__ void test_make_ulonglong3() {
  typedef cub::BlockReduce<ulonglong3, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: sycl::mulonglong3 res = sycl::reduce_over_group({{.+}}, sycl::mulonglong3(1., 2., 3.), {{.+}});
  ulonglong3 res = BlockReduce(smem_storage).Sum(make_ulonglong3(1., 2., 3.));
}

__device__ ulonglong4 operator+(ulonglong4 a, ulonglong4 b) {
  // CHECK: return sycl::mulonglong4(a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]);
  return make_ulonglong4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__global__ void test_make_ulonglong4() {
  typedef cub::BlockReduce<ulonglong4, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: sycl::mulonglong4 res = sycl::reduce_over_group({{.+}}, sycl::mulonglong4(1., 2., 3., 4.), {{.+}});
  ulonglong4 res = BlockReduce(smem_storage).Sum(make_ulonglong4(1., 2., 3., 4.));
}

__device__ float1 operator+(float1 a, float1 b) {
  // CHECK: return float(a + b);
  return make_float1(a.x + b.x);
}

__global__ void test_make_float1() {
  typedef cub::BlockReduce<float1, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: float res = sycl::reduce_over_group({{.+}}, float(1.), {{.+}});
  float1 res = BlockReduce(smem_storage).Sum(make_float1(1.));
}

__device__ float2 operator+(float2 a, float2 b) {
  // CHECK: return sycl::mfloat2(a[0] + b[0], a[1] + b[1]);
  return make_float2(a.x + b.x, a.y + b.y);
}

__global__ void test_make_float2() {
  typedef cub::BlockReduce<float2, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: sycl::mfloat2 res = sycl::reduce_over_group({{.+}}, sycl::mfloat2(1., 2.), {{.+}});
  float2 res = BlockReduce(smem_storage).Sum(make_float2(1., 2.));
}

__device__ float3 operator+(float3 a, float3 b) {
  // CHECK: return sycl::mfloat3(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__global__ void test_make_float3() {
  typedef cub::BlockReduce<float3, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: sycl::mfloat3 res = sycl::reduce_over_group({{.+}}, sycl::mfloat3(1., 2., 3.), {{.+}});
  float3 res = BlockReduce(smem_storage).Sum(make_float3(1., 2., 3.));
}

__device__ float4 operator+(float4 a, float4 b) {
  // CHECK: return sycl::mfloat4(a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]);
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__global__ void test_make_float4() {
  typedef cub::BlockReduce<float4, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: sycl::mfloat4 res = sycl::reduce_over_group({{.+}}, sycl::mfloat4(1., 2., 3., 4.), {{.+}});
  float4 res = BlockReduce(smem_storage).Sum(make_float4(1., 2., 3., 4.));
}

__device__ double1 operator+(double1 a, double1 b) {
  // CHECK: return double(a + b);
  return make_double1(a.x + b.x);
}

__global__ void test_make_double1() {
  typedef cub::BlockReduce<double1, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: double res = sycl::reduce_over_group({{.+}}, double(1.), {{.+}});
  double1 res = BlockReduce(smem_storage).Sum(make_double1(1.));
}

__device__ double2 operator+(double2 a, double2 b) {
  // CHECK: return sycl::mdouble2(a[0] + b[0], a[1] + b[1]);
  return make_double2(a.x + b.x, a.y + b.y);
}

__global__ void test_make_double2() {
  typedef cub::BlockReduce<double2, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: sycl::mdouble2 res = sycl::reduce_over_group({{.+}}, sycl::mdouble2(1., 2.), {{.+}});
  double2 res = BlockReduce(smem_storage).Sum(make_double2(1., 2.));
}

__device__ double3 operator+(double3 a, double3 b) {
  // CHECK: return sycl::mdouble3(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
  return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__global__ void test_make_double3() {
  typedef cub::BlockReduce<double3, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: sycl::mdouble3 res = sycl::reduce_over_group({{.+}}, sycl::mdouble3(1., 2., 3.), {{.+}});
  double3 res = BlockReduce(smem_storage).Sum(make_double3(1., 2., 3.));
}

__device__ double4 operator+(double4 a, double4 b) {
  // CHECK: return sycl::mdouble4(a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]);
  return make_double4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__global__ void test_make_double4() {
  typedef cub::BlockReduce<double4, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage smem_storage;
  // CHECK: sycl::mdouble4 res = sycl::reduce_over_group({{.+}}, sycl::mdouble4(1., 2., 3., 4.), {{.+}});
  double4 res = BlockReduce(smem_storage).Sum(make_double4(1., 2., 3., 4.));
}
