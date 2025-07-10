// REQUIRES: system-linux
// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1
// RUN: dpct --format-range=none -out-root %T/nvshmem %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/nvshmem/rma.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST %T/nvshmem/rma.dp.cpp -o %T/nvshmem/rma.dp.o %}
#include <nvshmem.h>
#include <nvshmemx.h>


__host__ __device__ void test(int target_pe) {
  const void *src_void;
  void *dst_void;

  // Standard RMA types
  float *src_float;
  float *dst_float;

  double *src_double;
  double *dst_double;

  char *src_char;
  char *dst_char;

  signed char *src_schar;
  signed char *dst_schar;

  short *src_short;
  short *dst_short;

  int *src_int;
  int *dst_int;

  long *src_long;
  long *dst_long;

  long long *src_longlong;
  long long *dst_longlong;

  unsigned char *src_uchar;
  unsigned char *dst_uchar;

  unsigned short *src_ushort;
  unsigned short *dst_ushort;

  unsigned int *src_uint;
  unsigned int *dst_uint;

  unsigned long *src_ulong;
  unsigned long *dst_ulong;

  unsigned long long *src_ulonglong;
  unsigned long long *dst_ulonglong;

  int8_t *src_int8;
  int8_t *dst_int8;

  int16_t *src_int16;
  int16_t *dst_int16;

  int32_t *src_int32;
  int32_t *dst_int32;

  int64_t *src_int64;
  int64_t *dst_int64;

  uint8_t *src_uint8;
  uint8_t *dst_uint8;

  uint16_t *src_uint16;
  uint16_t *dst_uint16;

  uint32_t *src_uint32;
  uint32_t *dst_uint32;

  uint64_t *src_uint64;
  uint64_t *dst_uint64;

  size_t *src_size;
  size_t *dst_size;

  ptrdiff_t *src_ptrdiff;
  ptrdiff_t *dst_ptrdiff;

  const int count = 10;

  // nvshmem_TYPENAME_put
  // ishmem_TYPENAME_put
  // CHECK: ishmem_put(dst_float, src_float, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_double, src_double, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_char, src_char, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_schar, src_schar, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_short, src_short, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_int, src_int, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_long, src_long, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_longlong, src_longlong, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_uchar, src_uchar, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_ushort, src_ushort, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_uint, src_uint, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_ulong, src_ulong, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_ulonglong, src_ulonglong, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_int8, src_int8, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_int16, src_int16, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_int32, src_int32, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_int64, src_int64, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_uint8, src_uint8, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_uint16, src_uint16, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_uint32, src_uint32, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_uint64, src_uint64, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_size, src_size, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_ptrdiff, src_ptrdiff, count, target_pe);
  nvshmem_float_put(dst_float, src_float, count, target_pe);
  nvshmem_double_put(dst_double, src_double, count, target_pe);
  nvshmem_char_put(dst_char, src_char, count, target_pe);
  nvshmem_schar_put(dst_schar, src_schar, count, target_pe);
  nvshmem_short_put(dst_short, src_short, count, target_pe);
  nvshmem_int_put(dst_int, src_int, count, target_pe);
  nvshmem_long_put(dst_long, src_long, count, target_pe);
  nvshmem_longlong_put(dst_longlong, src_longlong, count, target_pe);
  nvshmem_uchar_put(dst_uchar, src_uchar, count, target_pe);
  nvshmem_ushort_put(dst_ushort, src_ushort, count, target_pe);
  nvshmem_uint_put(dst_uint, src_uint, count, target_pe);
  nvshmem_ulong_put(dst_ulong, src_ulong, count, target_pe);
  nvshmem_ulonglong_put(dst_ulonglong, src_ulonglong, count, target_pe);
  nvshmem_int8_put(dst_int8, src_int8, count, target_pe);
  nvshmem_int16_put(dst_int16, src_int16, count, target_pe);
  nvshmem_int32_put(dst_int32, src_int32, count, target_pe);
  nvshmem_int64_put(dst_int64, src_int64, count, target_pe);
  nvshmem_uint8_put(dst_uint8, src_uint8, count, target_pe);
  nvshmem_uint16_put(dst_uint16, src_uint16, count, target_pe);
  nvshmem_uint32_put(dst_uint32, src_uint32, count, target_pe);
  nvshmem_uint64_put(dst_uint64, src_uint64, count, target_pe);
  nvshmem_size_put(dst_size, src_size, count, target_pe);
  nvshmem_ptrdiff_put(dst_ptrdiff, src_ptrdiff, count, target_pe);

  // nvshmemx_TYPENAME_put_block
  // ishmemx_TYPENAME_put_work_group
  // CHECK: ishmemx_put_work_group(dst_float, src_float, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_work_group(dst_double, src_double, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_work_group(dst_char, src_char, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_work_group(dst_schar, src_schar, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_work_group(dst_short, src_short, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_work_group(dst_int, src_int, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_work_group(dst_long, src_long, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_work_group(dst_longlong, src_longlong, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_work_group(dst_uchar, src_uchar, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_work_group(dst_ushort, src_ushort, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_work_group(dst_uint, src_uint, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_work_group(dst_ulong, src_ulong, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_work_group(dst_ulonglong, src_ulonglong, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_work_group(dst_int8, src_int8, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_work_group(dst_int16, src_int16, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_work_group(dst_int32, src_int32, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_work_group(dst_int64, src_int64, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_work_group(dst_uint8, src_uint8, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_work_group(dst_uint16, src_uint16, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_work_group(dst_uint32, src_uint32, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_work_group(dst_uint64, src_uint64, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_work_group(dst_size, src_size, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_work_group(dst_ptrdiff, src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  nvshmemx_float_put_block(dst_float, src_float, count, target_pe);
  nvshmemx_double_put_block(dst_double, src_double, count, target_pe);
  nvshmemx_char_put_block(dst_char, src_char, count, target_pe);
  nvshmemx_schar_put_block(dst_schar, src_schar, count, target_pe);
  nvshmemx_short_put_block(dst_short, src_short, count, target_pe);
  nvshmemx_int_put_block(dst_int, src_int, count, target_pe);
  nvshmemx_long_put_block(dst_long, src_long, count, target_pe);
  nvshmemx_longlong_put_block(dst_longlong, src_longlong, count, target_pe);
  nvshmemx_uchar_put_block(dst_uchar, src_uchar, count, target_pe);
  nvshmemx_ushort_put_block(dst_ushort, src_ushort, count, target_pe);
  nvshmemx_uint_put_block(dst_uint, src_uint, count, target_pe);
  nvshmemx_ulong_put_block(dst_ulong, src_ulong, count, target_pe);
  nvshmemx_ulonglong_put_block(dst_ulonglong, src_ulonglong, count, target_pe);
  nvshmemx_int8_put_block(dst_int8, src_int8, count, target_pe);
  nvshmemx_int16_put_block(dst_int16, src_int16, count, target_pe);
  nvshmemx_int32_put_block(dst_int32, src_int32, count, target_pe);
  nvshmemx_int64_put_block(dst_int64, src_int64, count, target_pe);
  nvshmemx_uint8_put_block(dst_uint8, src_uint8, count, target_pe);
  nvshmemx_uint16_put_block(dst_uint16, src_uint16, count, target_pe);
  nvshmemx_uint32_put_block(dst_uint32, src_uint32, count, target_pe);
  nvshmemx_uint64_put_block(dst_uint64, src_uint64, count, target_pe);
  nvshmemx_size_put_block(dst_size, src_size, count, target_pe);
  nvshmemx_ptrdiff_put_block(dst_ptrdiff, src_ptrdiff, count, target_pe);

  // nvshmemx_TYPENAME_put_warp
  // ishmemx_TYPENAME_put_work_group
  // CHECK: ishmemx_put_work_group(dst_float, src_float, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_work_group(dst_double, src_double, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_work_group(dst_char, src_char, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_work_group(dst_schar, src_schar, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_work_group(dst_short, src_short, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_work_group(dst_int, src_int, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_work_group(dst_long, src_long, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_work_group(dst_longlong, src_longlong, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_work_group(dst_uchar, src_uchar, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_work_group(dst_ushort, src_ushort, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_work_group(dst_uint, src_uint, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_work_group(dst_ulong, src_ulong, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_work_group(dst_ulonglong, src_ulonglong, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_work_group(dst_int8, src_int8, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_work_group(dst_int16, src_int16, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_work_group(dst_int32, src_int32, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_work_group(dst_int64, src_int64, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_work_group(dst_uint8, src_uint8, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_work_group(dst_uint16, src_uint16, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_work_group(dst_uint32, src_uint32, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_work_group(dst_uint64, src_uint64, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_work_group(dst_size, src_size, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_work_group(dst_ptrdiff, src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  nvshmemx_float_put_warp(dst_float, src_float, count, target_pe);
  nvshmemx_double_put_warp(dst_double, src_double, count, target_pe);
  nvshmemx_char_put_warp(dst_char, src_char, count, target_pe);
  nvshmemx_schar_put_warp(dst_schar, src_schar, count, target_pe);
  nvshmemx_short_put_warp(dst_short, src_short, count, target_pe);
  nvshmemx_int_put_warp(dst_int, src_int, count, target_pe);
  nvshmemx_long_put_warp(dst_long, src_long, count, target_pe);
  nvshmemx_longlong_put_warp(dst_longlong, src_longlong, count, target_pe);
  nvshmemx_uchar_put_warp(dst_uchar, src_uchar, count, target_pe);
  nvshmemx_ushort_put_warp(dst_ushort, src_ushort, count, target_pe);
  nvshmemx_uint_put_warp(dst_uint, src_uint, count, target_pe);
  nvshmemx_ulong_put_warp(dst_ulong, src_ulong, count, target_pe);
  nvshmemx_ulonglong_put_warp(dst_ulonglong, src_ulonglong, count, target_pe);
  nvshmemx_int8_put_warp(dst_int8, src_int8, count, target_pe);
  nvshmemx_int16_put_warp(dst_int16, src_int16, count, target_pe);
  nvshmemx_int32_put_warp(dst_int32, src_int32, count, target_pe);
  nvshmemx_int64_put_warp(dst_int64, src_int64, count, target_pe);
  nvshmemx_uint8_put_warp(dst_uint8, src_uint8, count, target_pe);
  nvshmemx_uint16_put_warp(dst_uint16, src_uint16, count, target_pe);
  nvshmemx_uint32_put_warp(dst_uint32, src_uint32, count, target_pe);
  nvshmemx_uint64_put_warp(dst_uint64, src_uint64, count, target_pe);
  nvshmemx_size_put_warp(dst_size, src_size, count, target_pe);
  nvshmemx_ptrdiff_put_warp(dst_ptrdiff, src_ptrdiff, count, target_pe);

  // nvshmem_putSIZE
  // ishmem_putSIZE
  // CHECK: ishmem_put8(dst_void, src_void, count, target_pe);
  // CHECK-NEXT: ishmem_put16(dst_void, src_void, count, target_pe);
  // CHECK-NEXT: ishmem_put32(dst_void, src_void, count, target_pe);
  // CHECK-NEXT: ishmem_put64(dst_void, src_void, count, target_pe);
  // CHECK-NEXT: ishmem_put128(dst_void, src_void, count, target_pe);
  nvshmem_put8(dst_void, src_void, count, target_pe);
  nvshmem_put16(dst_void, src_void, count, target_pe);
  nvshmem_put32(dst_void, src_void, count, target_pe);
  nvshmem_put64(dst_void, src_void, count, target_pe);
  nvshmem_put128(dst_void, src_void, count, target_pe);

  // nvshmemx_putSIZE_block
  // ishmemx_putSIZE_work_group
  // CHECK: ishmemx_put8_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put16_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put32_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put64_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put128_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  nvshmemx_put8_block(dst_void, src_void, count, target_pe);
  nvshmemx_put16_block(dst_void, src_void, count, target_pe);
  nvshmemx_put32_block(dst_void, src_void, count, target_pe);
  nvshmemx_put64_block(dst_void, src_void, count, target_pe);
  nvshmemx_put128_block(dst_void, src_void, count, target_pe);

  // nvshmemx_putSIZE_warp
  // ishmemx_putSIZE_work_group
  // CHECK: ishmemx_put8_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put16_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put32_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put64_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put128_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  nvshmemx_put8_warp(dst_void, src_void, count, target_pe);
  nvshmemx_put16_warp(dst_void, src_void, count, target_pe);
  nvshmemx_put32_warp(dst_void, src_void, count, target_pe);
  nvshmemx_put64_warp(dst_void, src_void, count, target_pe);
  nvshmemx_put128_warp(dst_void, src_void, count, target_pe);

  // nvshmem_TYPENAME_iput
  // ishmem_TYPENAME_iput
  // CHECK: ishmem_iput(dst_float, src_float, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_double, src_double, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_char, src_char, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_schar, src_schar, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_short, src_short, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_int, src_int, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_long, src_long, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_longlong, src_longlong, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_uchar, src_uchar, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_ushort, src_ushort, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_uint, src_uint, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_ulong, src_ulong, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_ulonglong, src_ulonglong, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_int8, src_int8, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_int16, src_int16, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_int32, src_int32, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_int64, src_int64, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_uint8, src_uint8, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_uint16, src_uint16, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_uint32, src_uint32, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_uint64, src_uint64, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_size, src_size, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_ptrdiff, src_ptrdiff, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_float_iput(dst_float, src_float, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_double_iput(dst_double, src_double, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_char_iput(dst_char, src_char, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_schar_iput(dst_schar, src_schar, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_short_iput(dst_short, src_short, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_int_iput(dst_int, src_int, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_long_iput(dst_long, src_long, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_longlong_iput(dst_longlong, src_longlong, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_uchar_iput(dst_uchar, src_uchar, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_ushort_iput(dst_ushort, src_ushort, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_uint_iput(dst_uint, src_uint, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_ulong_iput(dst_ulong, src_ulong, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_ulonglong_iput(dst_ulonglong, src_ulonglong, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_int8_iput(dst_int8, src_int8, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_int16_iput(dst_int16, src_int16, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_int32_iput(dst_int32, src_int32, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_int64_iput(dst_int64, src_int64, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_uint8_iput(dst_uint8, src_uint8, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_uint16_iput(dst_uint16, src_uint16, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_uint32_iput(dst_uint32, src_uint32, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_uint64_iput(dst_uint64, src_uint64, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_size_iput(dst_size, src_size, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_ptrdiff_iput(dst_ptrdiff, src_ptrdiff, *dst_ptrdiff, *src_ptrdiff, count, target_pe);

  // nvshmemx_TYPENAME_iput_block
  // ishmemx_TYPENAME_iput_work_group
  // CHECK: ishmemx_iput_work_group(dst_float, src_float, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_double, src_double, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_char, src_char, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_schar, src_schar, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_short, src_short, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_int, src_int, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_long, src_long, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_longlong, src_longlong, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_uchar, src_uchar, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_ushort, src_ushort, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_uint, src_uint, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_ulong, src_ulong, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_ulonglong, src_ulonglong, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_int8, src_int8, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_int16, src_int16, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_int32, src_int32, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_int64, src_int64, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_uint8, src_uint8, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_uint16, src_uint16, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_uint32, src_uint32, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_uint64, src_uint64, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_size, src_size, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_ptrdiff, src_ptrdiff, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  nvshmemx_float_iput_block(dst_float, src_float, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_double_iput_block(dst_double, src_double, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_char_iput_block(dst_char, src_char, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_schar_iput_block(dst_schar, src_schar, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_short_iput_block(dst_short, src_short, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_int_iput_block(dst_int, src_int, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_long_iput_block(dst_long, src_long, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_longlong_iput_block(dst_longlong, src_longlong, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_uchar_iput_block(dst_uchar, src_uchar, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_ushort_iput_block(dst_ushort, src_ushort, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_uint_iput_block(dst_uint, src_uint, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_ulong_iput_block(dst_ulong, src_ulong, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_ulonglong_iput_block(dst_ulonglong, src_ulonglong, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_int8_iput_block(dst_int8, src_int8, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_int16_iput_block(dst_int16, src_int16, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_int32_iput_block(dst_int32, src_int32, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_int64_iput_block(dst_int64, src_int64, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_uint8_iput_block(dst_uint8, src_uint8, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_uint16_iput_block(dst_uint16, src_uint16, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_uint32_iput_block(dst_uint32, src_uint32, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_uint64_iput_block(dst_uint64, src_uint64, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_size_iput_block(dst_size, src_size, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_ptrdiff_iput_block(dst_ptrdiff, src_ptrdiff, *dst_ptrdiff, *src_ptrdiff, count, target_pe);

  // nvshmemx_TYPENAME_iput_warp
  // ishmemx_TYPENAME_iput_work_group
  // CHECK: ishmemx_iput_work_group(dst_float, src_float, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_double, src_double, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_char, src_char, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_schar, src_schar, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_short, src_short, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_int, src_int, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_long, src_long, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_longlong, src_longlong, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_uchar, src_uchar, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_ushort, src_ushort, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_uint, src_uint, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_ulong, src_ulong, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_ulonglong, src_ulonglong, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_int8, src_int8, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_int16, src_int16, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_int32, src_int32, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_int64, src_int64, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_uint8, src_uint8, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_uint16, src_uint16, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_uint32, src_uint32, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_uint64, src_uint64, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_size, src_size, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iput_work_group(dst_ptrdiff, src_ptrdiff, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  nvshmemx_float_iput_warp(dst_float, src_float, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_double_iput_warp(dst_double, src_double, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_char_iput_warp(dst_char, src_char, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_schar_iput_warp(dst_schar, src_schar, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_short_iput_warp(dst_short, src_short, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_int_iput_warp(dst_int, src_int, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_long_iput_warp(dst_long, src_long, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_longlong_iput_warp(dst_longlong, src_longlong, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_uchar_iput_warp(dst_uchar, src_uchar, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_ushort_iput_warp(dst_ushort, src_ushort, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_uint_iput_warp(dst_uint, src_uint, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_ulong_iput_warp(dst_ulong, src_ulong, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_ulonglong_iput_warp(dst_ulonglong, src_ulonglong, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_int8_iput_warp(dst_int8, src_int8, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_int16_iput_warp(dst_int16, src_int16, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_int32_iput_warp(dst_int32, src_int32, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_int64_iput_warp(dst_int64, src_int64, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_uint8_iput_warp(dst_uint8, src_uint8, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_uint16_iput_warp(dst_uint16, src_uint16, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_uint32_iput_warp(dst_uint32, src_uint32, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_uint64_iput_warp(dst_uint64, src_uint64, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_size_iput_warp(dst_size, src_size, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_ptrdiff_iput_warp(dst_ptrdiff, src_ptrdiff, *dst_ptrdiff, *src_ptrdiff, count, target_pe);

  // nvshmem_iputSIZE
  // ishmem_iputSIZE
  // CHECK: ishmem_iput8(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput16(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput32(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput64(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput128(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_iput8(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_iput16(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_iput32(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_iput64(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_iput128(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);

  // nvshmemx_iputSIZE_block
  // ishmemx_iputSIZE_work_group
  // CHECK: ishmemx_iput8_work_group(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iput16_work_group(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iput32_work_group(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iput64_work_group(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iput128_work_group(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  nvshmemx_iput8_block(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_iput16_block(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_iput32_block(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_iput64_block(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_iput128_block(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);

  // nvshmemx_iputSIZE_warp
  // ishmemx_iputSIZE_work_group
  // CHECK: ishmemx_iput8_work_group(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iput16_work_group(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iput32_work_group(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iput64_work_group(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iput128_work_group(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  nvshmemx_iput8_warp(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_iput16_warp(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_iput32_warp(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_iput64_warp(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_iput128_warp(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);

  // nvshmem_putmem
  // ishmem_putmem
  // CHECK: ishmem_putmem(dst_void, src_void, count, target_pe);
  nvshmem_putmem(dst_void, src_void, count, target_pe);

  // nvshmemx_putmem_block
  // ishmemx_putmem_work_group
  // CHECK: ishmemx_putmem_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  nvshmemx_putmem_block(dst_void, src_void, count, target_pe);

  // nvshmemx_putmem_warp
  // ishmemx_putmem_work_group
  // CHECK: ishmemx_putmem_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  nvshmemx_putmem_warp(dst_void, src_void, count, target_pe);

  // nvshmem_TYPENAME_p
  // ishmem_TYPENAME_p
  // CHECK: ishmem_p(dst_float, *src_float, target_pe);
  // CHECK-NEXT: ishmem_p(dst_double, *src_double, target_pe);
  // CHECK-NEXT: ishmem_p(dst_char, *src_char, target_pe);
  // CHECK-NEXT: ishmem_p(dst_schar, *src_schar, target_pe);
  // CHECK-NEXT: ishmem_p(dst_short, *src_short, target_pe);
  // CHECK-NEXT: ishmem_p(dst_int, *src_int, target_pe);
  // CHECK-NEXT: ishmem_p(dst_long, *src_long, target_pe);
  // CHECK-NEXT: ishmem_p(dst_longlong, *src_longlong, target_pe);
  // CHECK-NEXT: ishmem_p(dst_uchar, *src_uchar, target_pe);
  // CHECK-NEXT: ishmem_p(dst_ushort, *src_ushort, target_pe);
  // CHECK-NEXT: ishmem_p(dst_uint, *src_uint, target_pe);
  // CHECK-NEXT: ishmem_p(dst_ulong, *src_ulong, target_pe);
  // CHECK-NEXT: ishmem_p(dst_ulonglong, *src_ulonglong, target_pe);
  // CHECK-NEXT: ishmem_p(dst_int8, *src_int8, target_pe);
  // CHECK-NEXT: ishmem_p(dst_int16, *src_int16, target_pe);
  // CHECK-NEXT: ishmem_p(dst_int32, *src_int32, target_pe);
  // CHECK-NEXT: ishmem_p(dst_int64, *src_int64, target_pe);
  // CHECK-NEXT: ishmem_p(dst_uint8, *src_uint8, target_pe);
  // CHECK-NEXT: ishmem_p(dst_uint16, *src_uint16, target_pe);
  // CHECK-NEXT: ishmem_p(dst_uint32, *src_uint32, target_pe);
  // CHECK-NEXT: ishmem_p(dst_uint64, *src_uint64, target_pe);
  // CHECK-NEXT: ishmem_p(dst_size, *src_size, target_pe);
  // CHECK-NEXT: ishmem_p(dst_ptrdiff, *src_ptrdiff, target_pe);
  nvshmem_float_p(dst_float, *src_float, target_pe);
  nvshmem_double_p(dst_double, *src_double, target_pe);
  nvshmem_char_p(dst_char, *src_char, target_pe);
  nvshmem_schar_p(dst_schar, *src_schar, target_pe);
  nvshmem_short_p(dst_short, *src_short, target_pe);
  nvshmem_int_p(dst_int, *src_int, target_pe);
  nvshmem_long_p(dst_long, *src_long, target_pe);
  nvshmem_longlong_p(dst_longlong, *src_longlong, target_pe);
  nvshmem_uchar_p(dst_uchar, *src_uchar, target_pe);
  nvshmem_ushort_p(dst_ushort, *src_ushort, target_pe);
  nvshmem_uint_p(dst_uint, *src_uint, target_pe);
  nvshmem_ulong_p(dst_ulong, *src_ulong, target_pe);
  nvshmem_ulonglong_p(dst_ulonglong, *src_ulonglong, target_pe);
  nvshmem_int8_p(dst_int8, *src_int8, target_pe);
  nvshmem_int16_p(dst_int16, *src_int16, target_pe);
  nvshmem_int32_p(dst_int32, *src_int32, target_pe);
  nvshmem_int64_p(dst_int64, *src_int64, target_pe);
  nvshmem_uint8_p(dst_uint8, *src_uint8, target_pe);
  nvshmem_uint16_p(dst_uint16, *src_uint16, target_pe);
  nvshmem_uint32_p(dst_uint32, *src_uint32, target_pe);
  nvshmem_uint64_p(dst_uint64, *src_uint64, target_pe);
  nvshmem_size_p(dst_size, *src_size, target_pe);
  nvshmem_ptrdiff_p(dst_ptrdiff, *src_ptrdiff, target_pe);

  // nvshmem_TYPENAME_get
  // ishmem_TYPENAME_get
  // CHECK: ishmem_get(dst_float, src_float, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_double, src_double, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_char, src_char, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_schar, src_schar, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_short, src_short, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_int, src_int, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_long, src_long, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_longlong, src_longlong, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_uchar, src_uchar, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_ushort, src_ushort, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_uint, src_uint, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_ulong, src_ulong, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_ulonglong, src_ulonglong, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_int8, src_int8, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_int16, src_int16, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_int32, src_int32, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_int64, src_int64, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_uint8, src_uint8, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_uint16, src_uint16, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_uint32, src_uint32, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_uint64, src_uint64, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_size, src_size, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_ptrdiff, src_ptrdiff, count, target_pe);
  nvshmem_float_get(dst_float, src_float, count, target_pe);
  nvshmem_double_get(dst_double, src_double, count, target_pe);
  nvshmem_char_get(dst_char, src_char, count, target_pe);
  nvshmem_schar_get(dst_schar, src_schar, count, target_pe);
  nvshmem_short_get(dst_short, src_short, count, target_pe);
  nvshmem_int_get(dst_int, src_int, count, target_pe);
  nvshmem_long_get(dst_long, src_long, count, target_pe);
  nvshmem_longlong_get(dst_longlong, src_longlong, count, target_pe);
  nvshmem_uchar_get(dst_uchar, src_uchar, count, target_pe);
  nvshmem_ushort_get(dst_ushort, src_ushort, count, target_pe);
  nvshmem_uint_get(dst_uint, src_uint, count, target_pe);
  nvshmem_ulong_get(dst_ulong, src_ulong, count, target_pe);
  nvshmem_ulonglong_get(dst_ulonglong, src_ulonglong, count, target_pe);
  nvshmem_int8_get(dst_int8, src_int8, count, target_pe);
  nvshmem_int16_get(dst_int16, src_int16, count, target_pe);
  nvshmem_int32_get(dst_int32, src_int32, count, target_pe);
  nvshmem_int64_get(dst_int64, src_int64, count, target_pe);
  nvshmem_uint8_get(dst_uint8, src_uint8, count, target_pe);
  nvshmem_uint16_get(dst_uint16, src_uint16, count, target_pe);
  nvshmem_uint32_get(dst_uint32, src_uint32, count, target_pe);
  nvshmem_uint64_get(dst_uint64, src_uint64, count, target_pe);
  nvshmem_size_get(dst_size, src_size, count, target_pe);
  nvshmem_ptrdiff_get(dst_ptrdiff, src_ptrdiff, count, target_pe);

  // nvshmemx_TYPENAME_get_block
  // ishmemx_TYPENAME_get_work_group
  // CHECK: ishmemx_get_work_group(dst_float, src_float, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_work_group(dst_double, src_double, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_work_group(dst_char, src_char, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_work_group(dst_schar, src_schar, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_work_group(dst_short, src_short, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_work_group(dst_int, src_int, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_work_group(dst_long, src_long, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_work_group(dst_longlong, src_longlong, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_work_group(dst_uchar, src_uchar, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_work_group(dst_ushort, src_ushort, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_work_group(dst_uint, src_uint, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_work_group(dst_ulong, src_ulong, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_work_group(dst_ulonglong, src_ulonglong, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_work_group(dst_int8, src_int8, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_work_group(dst_int16, src_int16, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_work_group(dst_int32, src_int32, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_work_group(dst_int64, src_int64, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_work_group(dst_uint8, src_uint8, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_work_group(dst_uint16, src_uint16, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_work_group(dst_uint32, src_uint32, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_work_group(dst_uint64, src_uint64, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_work_group(dst_size, src_size, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_work_group(dst_ptrdiff, src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  nvshmemx_float_get_block(dst_float, src_float, count, target_pe);
  nvshmemx_double_get_block(dst_double, src_double, count, target_pe);
  nvshmemx_char_get_block(dst_char, src_char, count, target_pe);
  nvshmemx_schar_get_block(dst_schar, src_schar, count, target_pe);
  nvshmemx_short_get_block(dst_short, src_short, count, target_pe);
  nvshmemx_int_get_block(dst_int, src_int, count, target_pe);
  nvshmemx_long_get_block(dst_long, src_long, count, target_pe);
  nvshmemx_longlong_get_block(dst_longlong, src_longlong, count, target_pe);
  nvshmemx_uchar_get_block(dst_uchar, src_uchar, count, target_pe);
  nvshmemx_ushort_get_block(dst_ushort, src_ushort, count, target_pe);
  nvshmemx_uint_get_block(dst_uint, src_uint, count, target_pe);
  nvshmemx_ulong_get_block(dst_ulong, src_ulong, count, target_pe);
  nvshmemx_ulonglong_get_block(dst_ulonglong, src_ulonglong, count, target_pe);
  nvshmemx_int8_get_block(dst_int8, src_int8, count, target_pe);
  nvshmemx_int16_get_block(dst_int16, src_int16, count, target_pe);
  nvshmemx_int32_get_block(dst_int32, src_int32, count, target_pe);
  nvshmemx_int64_get_block(dst_int64, src_int64, count, target_pe);
  nvshmemx_uint8_get_block(dst_uint8, src_uint8, count, target_pe);
  nvshmemx_uint16_get_block(dst_uint16, src_uint16, count, target_pe);
  nvshmemx_uint32_get_block(dst_uint32, src_uint32, count, target_pe);
  nvshmemx_uint64_get_block(dst_uint64, src_uint64, count, target_pe);
  nvshmemx_size_get_block(dst_size, src_size, count, target_pe);
  nvshmemx_ptrdiff_get_block(dst_ptrdiff, src_ptrdiff, count, target_pe);

  // nvshmemx_TYPENAME_get_warp
  // ishmemx_TYPENAME_get_work_group
  // CHECK: ishmemx_get_work_group(dst_float, src_float, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_work_group(dst_double, src_double, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_work_group(dst_char, src_char, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_work_group(dst_schar, src_schar, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_work_group(dst_short, src_short, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_work_group(dst_int, src_int, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_work_group(dst_long, src_long, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_work_group(dst_longlong, src_longlong, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_work_group(dst_uchar, src_uchar, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_work_group(dst_ushort, src_ushort, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_work_group(dst_uint, src_uint, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_work_group(dst_ulong, src_ulong, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_work_group(dst_ulonglong, src_ulonglong, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_work_group(dst_int8, src_int8, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_work_group(dst_int16, src_int16, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_work_group(dst_int32, src_int32, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_work_group(dst_int64, src_int64, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_work_group(dst_uint8, src_uint8, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_work_group(dst_uint16, src_uint16, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_work_group(dst_uint32, src_uint32, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_work_group(dst_uint64, src_uint64, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_work_group(dst_size, src_size, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_work_group(dst_ptrdiff, src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  nvshmemx_float_get_warp(dst_float, src_float, count, target_pe);
  nvshmemx_double_get_warp(dst_double, src_double, count, target_pe);
  nvshmemx_char_get_warp(dst_char, src_char, count, target_pe);
  nvshmemx_schar_get_warp(dst_schar, src_schar, count, target_pe);
  nvshmemx_short_get_warp(dst_short, src_short, count, target_pe);
  nvshmemx_int_get_warp(dst_int, src_int, count, target_pe);
  nvshmemx_long_get_warp(dst_long, src_long, count, target_pe);
  nvshmemx_longlong_get_warp(dst_longlong, src_longlong, count, target_pe);
  nvshmemx_uchar_get_warp(dst_uchar, src_uchar, count, target_pe);
  nvshmemx_ushort_get_warp(dst_ushort, src_ushort, count, target_pe);
  nvshmemx_uint_get_warp(dst_uint, src_uint, count, target_pe);
  nvshmemx_ulong_get_warp(dst_ulong, src_ulong, count, target_pe);
  nvshmemx_ulonglong_get_warp(dst_ulonglong, src_ulonglong, count, target_pe);
  nvshmemx_int8_get_warp(dst_int8, src_int8, count, target_pe);
  nvshmemx_int16_get_warp(dst_int16, src_int16, count, target_pe);
  nvshmemx_int32_get_warp(dst_int32, src_int32, count, target_pe);
  nvshmemx_int64_get_warp(dst_int64, src_int64, count, target_pe);
  nvshmemx_uint8_get_warp(dst_uint8, src_uint8, count, target_pe);
  nvshmemx_uint16_get_warp(dst_uint16, src_uint16, count, target_pe);
  nvshmemx_uint32_get_warp(dst_uint32, src_uint32, count, target_pe);
  nvshmemx_uint64_get_warp(dst_uint64, src_uint64, count, target_pe);
  nvshmemx_size_get_warp(dst_size, src_size, count, target_pe);
  nvshmemx_ptrdiff_get_warp(dst_ptrdiff, src_ptrdiff, count, target_pe);

  // nvshmem_getSIZE
  // ishmem_getSIZE
  // CHECK: ishmem_get8(dst_void, src_void, count, target_pe);
  // CHECK-NEXT: ishmem_get16(dst_void, src_void, count, target_pe);
  // CHECK-NEXT: ishmem_get32(dst_void, src_void, count, target_pe);
  // CHECK-NEXT: ishmem_get64(dst_void, src_void, count, target_pe);
  // CHECK-NEXT: ishmem_get128(dst_void, src_void, count, target_pe);
  nvshmem_get8(dst_void, src_void, count, target_pe);
  nvshmem_get16(dst_void, src_void, count, target_pe);
  nvshmem_get32(dst_void, src_void, count, target_pe);
  nvshmem_get64(dst_void, src_void, count, target_pe);
  nvshmem_get128(dst_void, src_void, count, target_pe);

  // nvshmemx_getSIZE_block
  // ishmemx_getSIZE_work_group
  // CHECK: ishmemx_get8_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get16_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get32_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get64_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get128_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  nvshmemx_get8_block(dst_void, src_void, count, target_pe);
  nvshmemx_get16_block(dst_void, src_void, count, target_pe);
  nvshmemx_get32_block(dst_void, src_void, count, target_pe);
  nvshmemx_get64_block(dst_void, src_void, count, target_pe);
  nvshmemx_get128_block(dst_void, src_void, count, target_pe);

  // nvshmemx_getSIZE_warp
  // ishmemx_getSIZE_work_group
  // CHECK: ishmemx_get8_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get16_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get32_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get64_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get128_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  nvshmemx_get8_warp(dst_void, src_void, count, target_pe);
  nvshmemx_get16_warp(dst_void, src_void, count, target_pe);
  nvshmemx_get32_warp(dst_void, src_void, count, target_pe);
  nvshmemx_get64_warp(dst_void, src_void, count, target_pe);
  nvshmemx_get128_warp(dst_void, src_void, count, target_pe);

  // nvshmem_TYPENAME_iget
  // ishmem_TYPENAME_iget
  // CHECK: ishmem_iget(dst_float, src_float, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_double, src_double, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_char, src_char, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_schar, src_schar, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_short, src_short, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_int, src_int, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_long, src_long, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_longlong, src_longlong, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_uchar, src_uchar, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_ushort, src_ushort, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_uint, src_uint, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_ulong, src_ulong, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_ulonglong, src_ulonglong, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_int8, src_int8, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_int16, src_int16, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_int32, src_int32, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_int64, src_int64, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_uint8, src_uint8, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_uint16, src_uint16, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_uint32, src_uint32, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_uint64, src_uint64, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_size, src_size, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_ptrdiff, src_ptrdiff, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_float_iget(dst_float, src_float, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_double_iget(dst_double, src_double, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_char_iget(dst_char, src_char, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_schar_iget(dst_schar, src_schar, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_short_iget(dst_short, src_short, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_int_iget(dst_int, src_int, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_long_iget(dst_long, src_long, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_longlong_iget(dst_longlong, src_longlong, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_uchar_iget(dst_uchar, src_uchar, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_ushort_iget(dst_ushort, src_ushort, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_uint_iget(dst_uint, src_uint, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_ulong_iget(dst_ulong, src_ulong, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_ulonglong_iget(dst_ulonglong, src_ulonglong, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_int8_iget(dst_int8, src_int8, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_int16_iget(dst_int16, src_int16, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_int32_iget(dst_int32, src_int32, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_int64_iget(dst_int64, src_int64, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_uint8_iget(dst_uint8, src_uint8, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_uint16_iget(dst_uint16, src_uint16, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_uint32_iget(dst_uint32, src_uint32, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_uint64_iget(dst_uint64, src_uint64, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_size_iget(dst_size, src_size, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_ptrdiff_iget(dst_ptrdiff, src_ptrdiff, *dst_ptrdiff, *src_ptrdiff, count, target_pe);

  // nvshmemx_TYPENAME_iget_block
  // ishmemx_TYPENAME_iget_work_group
  // CHECK: ishmemx_iget_work_group(dst_float, src_float, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_double, src_double, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_char, src_char, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_schar, src_schar, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_short, src_short, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_int, src_int, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_long, src_long, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_longlong, src_longlong, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_uchar, src_uchar, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_ushort, src_ushort, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_uint, src_uint, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_ulong, src_ulong, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_ulonglong, src_ulonglong, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_int8, src_int8, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_int16, src_int16, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_int32, src_int32, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_int64, src_int64, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_uint8, src_uint8, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_uint16, src_uint16, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_uint32, src_uint32, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_uint64, src_uint64, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_size, src_size, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_ptrdiff, src_ptrdiff, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  nvshmemx_float_iget_block(dst_float, src_float, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_double_iget_block(dst_double, src_double, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_char_iget_block(dst_char, src_char, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_schar_iget_block(dst_schar, src_schar, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_short_iget_block(dst_short, src_short, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_int_iget_block(dst_int, src_int, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_long_iget_block(dst_long, src_long, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_longlong_iget_block(dst_longlong, src_longlong, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_uchar_iget_block(dst_uchar, src_uchar, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_ushort_iget_block(dst_ushort, src_ushort, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_uint_iget_block(dst_uint, src_uint, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_ulong_iget_block(dst_ulong, src_ulong, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_ulonglong_iget_block(dst_ulonglong, src_ulonglong, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_int8_iget_block(dst_int8, src_int8, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_int16_iget_block(dst_int16, src_int16, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_int32_iget_block(dst_int32, src_int32, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_int64_iget_block(dst_int64, src_int64, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_uint8_iget_block(dst_uint8, src_uint8, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_uint16_iget_block(dst_uint16, src_uint16, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_uint32_iget_block(dst_uint32, src_uint32, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_uint64_iget_block(dst_uint64, src_uint64, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_size_iget_block(dst_size, src_size, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_ptrdiff_iget_block(dst_ptrdiff, src_ptrdiff, *dst_ptrdiff, *src_ptrdiff, count, target_pe);

  // nvshmemx_TYPENAME_iget_warp
  // ishmemx_TYPENAME_iget_work_group
  // CHECK: ishmemx_iget_work_group(dst_float, src_float, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_double, src_double, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_char, src_char, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_schar, src_schar, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_short, src_short, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_int, src_int, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_long, src_long, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_longlong, src_longlong, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_uchar, src_uchar, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_ushort, src_ushort, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_uint, src_uint, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_ulong, src_ulong, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_ulonglong, src_ulonglong, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_int8, src_int8, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_int16, src_int16, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_int32, src_int32, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_int64, src_int64, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_uint8, src_uint8, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_uint16, src_uint16, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_uint32, src_uint32, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_uint64, src_uint64, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_size, src_size, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iget_work_group(dst_ptrdiff, src_ptrdiff, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  nvshmemx_float_iget_warp(dst_float, src_float, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_double_iget_warp(dst_double, src_double, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_char_iget_warp(dst_char, src_char, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_schar_iget_warp(dst_schar, src_schar, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_short_iget_warp(dst_short, src_short, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_int_iget_warp(dst_int, src_int, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_long_iget_warp(dst_long, src_long, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_longlong_iget_warp(dst_longlong, src_longlong, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_uchar_iget_warp(dst_uchar, src_uchar, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_ushort_iget_warp(dst_ushort, src_ushort, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_uint_iget_warp(dst_uint, src_uint, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_ulong_iget_warp(dst_ulong, src_ulong, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_ulonglong_iget_warp(dst_ulonglong, src_ulonglong, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_int8_iget_warp(dst_int8, src_int8, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_int16_iget_warp(dst_int16, src_int16, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_int32_iget_warp(dst_int32, src_int32, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_int64_iget_warp(dst_int64, src_int64, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_uint8_iget_warp(dst_uint8, src_uint8, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_uint16_iget_warp(dst_uint16, src_uint16, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_uint32_iget_warp(dst_uint32, src_uint32, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_uint64_iget_warp(dst_uint64, src_uint64, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_size_iget_warp(dst_size, src_size, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_ptrdiff_iget_warp(dst_ptrdiff, src_ptrdiff, *dst_ptrdiff, *src_ptrdiff, count, target_pe);

  // nvshmem_igetSIZE
  // ishmem_igetSIZE
  // CHECK: ishmem_iget8(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget16(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget32(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget64(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget128(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_iget8(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_iget16(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_iget32(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_iget64(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_iget128(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);

  // nvshmemx_igetSIZE_block
  // ishmemx_igetSIZE_work_group
  // CHECK: ishmemx_iget8_work_group(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iget16_work_group(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iget32_work_group(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iget64_work_group(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_iget128_work_group(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  nvshmemx_iget8_block(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_iget16_block(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_iget32_block(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_iget64_block(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_iget128_block(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);

  // nvshmemx_igetSIZE_warp
  // ishmemx_igetSIZE_work_group
  // CHECK: ishmemx_iget8_work_group(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iget16_work_group(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iget32_work_group(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iget64_work_group(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_iget128_work_group(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  nvshmemx_iget8_warp(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_iget16_warp(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_iget32_warp(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_iget64_warp(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmemx_iget128_warp(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);

  // nvshmem_getmem
  // ishmem_getmem
  // CHECK: ishmem_getmem(dst_void, src_void, count, target_pe);
  nvshmem_getmem(dst_void, src_void, count, target_pe);

  // nvshmemx_getmem_block
  // ishmemx_getmem_work_group
  // CHECK: ishmemx_getmem_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  nvshmemx_getmem_block(dst_void, src_void, count, target_pe);

  // nvshmemx_getmem_warp
  // ishmemx_getmem_work_group
  // CHECK: ishmemx_getmem_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  nvshmemx_getmem_warp(dst_void, src_void, count, target_pe);

  // nvshmem_TYPENAME_g
  // ishmem_TYPENAME_g
  // CHECK: *dst_float = ishmem_g(src_float, target_pe);
  // CHECK-NEXT: *dst_double = ishmem_g(src_double, target_pe);
  // CHECK-NEXT: *dst_char = ishmem_g(src_char, target_pe);
  // CHECK-NEXT: *dst_schar = ishmem_g(src_schar, target_pe);
  // CHECK-NEXT: *dst_short = ishmem_g(src_short, target_pe);
  // CHECK-NEXT: *dst_int = ishmem_g(src_int, target_pe);
  // CHECK-NEXT: *dst_long = ishmem_g(src_long, target_pe);
  // CHECK-NEXT: *dst_longlong = ishmem_g(src_longlong, target_pe);
  // CHECK-NEXT: *dst_uchar = ishmem_g(src_uchar, target_pe);
  // CHECK-NEXT: *dst_ushort = ishmem_g(src_ushort, target_pe);
  // CHECK-NEXT: *dst_uint = ishmem_g(src_uint, target_pe);
  // CHECK-NEXT: *dst_ulong = ishmem_g(src_ulong, target_pe);
  // CHECK-NEXT: *dst_ulonglong = ishmem_g(src_ulonglong, target_pe);
  // CHECK-NEXT: *dst_int8 = ishmem_g(src_int8, target_pe);
  // CHECK-NEXT: *dst_int16 = ishmem_g(src_int16, target_pe);
  // CHECK-NEXT: *dst_int32 = ishmem_g(src_int32, target_pe);
  // CHECK-NEXT: *dst_int64 = ishmem_g(src_int64, target_pe);
  // CHECK-NEXT: *dst_uint8 = ishmem_g(src_uint8, target_pe);
  // CHECK-NEXT: *dst_uint16 = ishmem_g(src_uint16, target_pe);
  // CHECK-NEXT: *dst_uint32 = ishmem_g(src_uint32, target_pe);
  // CHECK-NEXT: *dst_uint64 = ishmem_g(src_uint64, target_pe);
  // CHECK-NEXT: *dst_size = ishmem_g(src_size, target_pe);
  // CHECK-NEXT: *dst_ptrdiff = ishmem_g(src_ptrdiff, target_pe);
  *dst_float = nvshmem_float_g(src_float, target_pe);
  *dst_double = nvshmem_double_g(src_double, target_pe);
  *dst_char = nvshmem_char_g(src_char, target_pe);
  *dst_schar = nvshmem_schar_g(src_schar, target_pe);
  *dst_short = nvshmem_short_g(src_short, target_pe);
  *dst_int = nvshmem_int_g(src_int, target_pe);
  *dst_long = nvshmem_long_g(src_long, target_pe);
  *dst_longlong = nvshmem_longlong_g(src_longlong, target_pe);
  *dst_uchar = nvshmem_uchar_g(src_uchar, target_pe);
  *dst_ushort = nvshmem_ushort_g(src_ushort, target_pe);
  *dst_uint = nvshmem_uint_g(src_uint, target_pe);
  *dst_ulong = nvshmem_ulong_g(src_ulong, target_pe);
  *dst_ulonglong = nvshmem_ulonglong_g(src_ulonglong, target_pe);
  *dst_int8 = nvshmem_int8_g(src_int8, target_pe);
  *dst_int16 = nvshmem_int16_g(src_int16, target_pe);
  *dst_int32 = nvshmem_int32_g(src_int32, target_pe);
  *dst_int64 = nvshmem_int64_g(src_int64, target_pe);
  *dst_uint8 = nvshmem_uint8_g(src_uint8, target_pe);
  *dst_uint16 = nvshmem_uint16_g(src_uint16, target_pe);
  *dst_uint32 = nvshmem_uint32_g(src_uint32, target_pe);
  *dst_uint64 = nvshmem_uint64_g(src_uint64, target_pe);
  *dst_size = nvshmem_size_g(src_size, target_pe);
  *dst_ptrdiff = nvshmem_ptrdiff_g(src_ptrdiff, target_pe);
}


int main() {
  const void *src_void;
  void *dst_void;

  // Standard RMA types
  float *src_float;
  float *dst_float;

  double *src_double;
  double *dst_double;

  char *src_char;
  char *dst_char;

  signed char *src_schar;
  signed char *dst_schar;

  short *src_short;
  short *dst_short;

  int *src_int;
  int *dst_int;

  long *src_long;
  long *dst_long;

  long long *src_longlong;
  long long *dst_longlong;

  unsigned char *src_uchar;
  unsigned char *dst_uchar;

  unsigned short *src_ushort;
  unsigned short *dst_ushort;

  unsigned int *src_uint;
  unsigned int *dst_uint;

  unsigned long *src_ulong;
  unsigned long *dst_ulong;

  unsigned long long *src_ulonglong;
  unsigned long long *dst_ulonglong;

  int8_t *src_int8;
  int8_t *dst_int8;

  int16_t *src_int16;
  int16_t *dst_int16;

  int32_t *src_int32;
  int32_t *dst_int32;

  int64_t *src_int64;
  int64_t *dst_int64;

  uint8_t *src_uint8;
  uint8_t *dst_uint8;

  uint16_t *src_uint16;
  uint16_t *dst_uint16;

  uint32_t *src_uint32;
  uint32_t *dst_uint32;

  uint64_t *src_uint64;
  uint64_t *dst_uint64;

  size_t *src_size;
  size_t *dst_size;

  ptrdiff_t *src_ptrdiff;
  ptrdiff_t *dst_ptrdiff;

  int target_pe = 0;
  const int count = 10;
  cudaStream_t stream = 0;

  // nvshmem_TYPENAME_put
  // ishmem_TYPENAME_put
  // CHECK: ishmem_put(dst_float, src_float, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_double, src_double, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_char, src_char, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_schar, src_schar, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_short, src_short, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_int, src_int, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_long, src_long, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_longlong, src_longlong, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_uchar, src_uchar, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_ushort, src_ushort, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_uint, src_uint, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_ulong, src_ulong, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_ulonglong, src_ulonglong, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_int8, src_int8, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_int16, src_int16, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_int32, src_int32, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_int64, src_int64, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_uint8, src_uint8, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_uint16, src_uint16, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_uint32, src_uint32, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_uint64, src_uint64, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_size, src_size, count, target_pe);
  // CHECK-NEXT: ishmem_put(dst_ptrdiff, src_ptrdiff, count, target_pe);
  nvshmem_float_put(dst_float, src_float, count, target_pe);
  nvshmem_double_put(dst_double, src_double, count, target_pe);
  nvshmem_char_put(dst_char, src_char, count, target_pe);
  nvshmem_schar_put(dst_schar, src_schar, count, target_pe);
  nvshmem_short_put(dst_short, src_short, count, target_pe);
  nvshmem_int_put(dst_int, src_int, count, target_pe);
  nvshmem_long_put(dst_long, src_long, count, target_pe);
  nvshmem_longlong_put(dst_longlong, src_longlong, count, target_pe);
  nvshmem_uchar_put(dst_uchar, src_uchar, count, target_pe);
  nvshmem_ushort_put(dst_ushort, src_ushort, count, target_pe);
  nvshmem_uint_put(dst_uint, src_uint, count, target_pe);
  nvshmem_ulong_put(dst_ulong, src_ulong, count, target_pe);
  nvshmem_ulonglong_put(dst_ulonglong, src_ulonglong, count, target_pe);
  nvshmem_int8_put(dst_int8, src_int8, count, target_pe);
  nvshmem_int16_put(dst_int16, src_int16, count, target_pe);
  nvshmem_int32_put(dst_int32, src_int32, count, target_pe);
  nvshmem_int64_put(dst_int64, src_int64, count, target_pe);
  nvshmem_uint8_put(dst_uint8, src_uint8, count, target_pe);
  nvshmem_uint16_put(dst_uint16, src_uint16, count, target_pe);
  nvshmem_uint32_put(dst_uint32, src_uint32, count, target_pe);
  nvshmem_uint64_put(dst_uint64, src_uint64, count, target_pe);
  nvshmem_size_put(dst_size, src_size, count, target_pe);
  nvshmem_ptrdiff_put(dst_ptrdiff, src_ptrdiff, count, target_pe);

  // nvshmemx_TYPENAME_put_on_stream
  // ishmemx_TYPENAME_put_on_queue
  // CHECK: ishmemx_put_on_queue(dst_float, src_float, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_on_queue(dst_double, src_double, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_on_queue(dst_char, src_char, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_on_queue(dst_schar, src_schar, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_on_queue(dst_short, src_short, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_on_queue(dst_int, src_int, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_on_queue(dst_long, src_long, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_on_queue(dst_longlong, src_longlong, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_on_queue(dst_uchar, src_uchar, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_on_queue(dst_ushort, src_ushort, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_on_queue(dst_uint, src_uint, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_on_queue(dst_ulong, src_ulong, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_on_queue(dst_ulonglong, src_ulonglong, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_on_queue(dst_int8, src_int8, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_on_queue(dst_int16, src_int16, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_on_queue(dst_int32, src_int32, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_on_queue(dst_int64, src_int64, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_on_queue(dst_uint8, src_uint8, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_on_queue(dst_uint16, src_uint16, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_on_queue(dst_uint32, src_uint32, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_on_queue(dst_uint64, src_uint64, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_on_queue(dst_size, src_size, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_on_queue(dst_ptrdiff, src_ptrdiff, count, target_pe, *stream);
  nvshmemx_float_put_on_stream(dst_float, src_float, count, target_pe, stream);
  nvshmemx_double_put_on_stream(dst_double, src_double, count, target_pe, stream);
  nvshmemx_char_put_on_stream(dst_char, src_char, count, target_pe, stream);
  nvshmemx_schar_put_on_stream(dst_schar, src_schar, count, target_pe, stream);
  nvshmemx_short_put_on_stream(dst_short, src_short, count, target_pe, stream);
  nvshmemx_int_put_on_stream(dst_int, src_int, count, target_pe, stream);
  nvshmemx_long_put_on_stream(dst_long, src_long, count, target_pe, stream);
  nvshmemx_longlong_put_on_stream(dst_longlong, src_longlong, count, target_pe, stream);
  nvshmemx_uchar_put_on_stream(dst_uchar, src_uchar, count, target_pe, stream);
  nvshmemx_ushort_put_on_stream(dst_ushort, src_ushort, count, target_pe, stream);
  nvshmemx_uint_put_on_stream(dst_uint, src_uint, count, target_pe, stream);
  nvshmemx_ulong_put_on_stream(dst_ulong, src_ulong, count, target_pe, stream);
  nvshmemx_ulonglong_put_on_stream(dst_ulonglong, src_ulonglong, count, target_pe, stream);
  nvshmemx_int8_put_on_stream(dst_int8, src_int8, count, target_pe, stream);
  nvshmemx_int16_put_on_stream(dst_int16, src_int16, count, target_pe, stream);
  nvshmemx_int32_put_on_stream(dst_int32, src_int32, count, target_pe, stream);
  nvshmemx_int64_put_on_stream(dst_int64, src_int64, count, target_pe, stream);
  nvshmemx_uint8_put_on_stream(dst_uint8, src_uint8, count, target_pe, stream);
  nvshmemx_uint16_put_on_stream(dst_uint16, src_uint16, count, target_pe, stream);
  nvshmemx_uint32_put_on_stream(dst_uint32, src_uint32, count, target_pe, stream);
  nvshmemx_uint64_put_on_stream(dst_uint64, src_uint64, count, target_pe, stream);
  nvshmemx_size_put_on_stream(dst_size, src_size, count, target_pe, stream);
  nvshmemx_ptrdiff_put_on_stream(dst_ptrdiff, src_ptrdiff, count, target_pe, stream);

  // nvshmem_putSIZE
  // ishmem_putSIZE
  // CHECK: ishmem_put8(dst_void, src_void, count, target_pe);
  // CHECK-NEXT: ishmem_put16(dst_void, src_void, count, target_pe);
  // CHECK-NEXT: ishmem_put32(dst_void, src_void, count, target_pe);
  // CHECK-NEXT: ishmem_put64(dst_void, src_void, count, target_pe);
  // CHECK-NEXT: ishmem_put128(dst_void, src_void, count, target_pe);
  nvshmem_put8(dst_void, src_void, count, target_pe);
  nvshmem_put16(dst_void, src_void, count, target_pe);
  nvshmem_put32(dst_void, src_void, count, target_pe);
  nvshmem_put64(dst_void, src_void, count, target_pe);
  nvshmem_put128(dst_void, src_void, count, target_pe);

  // nvshmemx_putSIZE_on_stream
  // ishmemx_putSIZE_on_queue
  // CHECK: ishmemx_put8_on_queue(dst_void, src_void, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put16_on_queue(dst_void, src_void, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put32_on_queue(dst_void, src_void, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put64_on_queue(dst_void, src_void, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put128_on_queue(dst_void, src_void, count, target_pe, *stream);
  nvshmemx_put8_on_stream(dst_void, src_void, count, target_pe, stream);
  nvshmemx_put16_on_stream(dst_void, src_void, count, target_pe, stream);
  nvshmemx_put32_on_stream(dst_void, src_void, count, target_pe, stream);
  nvshmemx_put64_on_stream(dst_void, src_void, count, target_pe, stream);
  nvshmemx_put128_on_stream(dst_void, src_void, count, target_pe, stream);

  // nvshmem_TYPENAME_iput
  // ishmem_TYPENAME_iput
  // CHECK: ishmem_iput(dst_float, src_float, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_double, src_double, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_char, src_char, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_schar, src_schar, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_short, src_short, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_int, src_int, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_long, src_long, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_longlong, src_longlong, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_uchar, src_uchar, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_ushort, src_ushort, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_uint, src_uint, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_ulong, src_ulong, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_ulonglong, src_ulonglong, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_int8, src_int8, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_int16, src_int16, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_int32, src_int32, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_int64, src_int64, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_uint8, src_uint8, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_uint16, src_uint16, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_uint32, src_uint32, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_uint64, src_uint64, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_size, src_size, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput(dst_ptrdiff, src_ptrdiff, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_float_iput(dst_float, src_float, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_double_iput(dst_double, src_double, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_char_iput(dst_char, src_char, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_schar_iput(dst_schar, src_schar, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_short_iput(dst_short, src_short, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_int_iput(dst_int, src_int, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_long_iput(dst_long, src_long, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_longlong_iput(dst_longlong, src_longlong, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_uchar_iput(dst_uchar, src_uchar, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_ushort_iput(dst_ushort, src_ushort, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_uint_iput(dst_uint, src_uint, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_ulong_iput(dst_ulong, src_ulong, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_ulonglong_iput(dst_ulonglong, src_ulonglong, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_int8_iput(dst_int8, src_int8, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_int16_iput(dst_int16, src_int16, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_int32_iput(dst_int32, src_int32, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_int64_iput(dst_int64, src_int64, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_uint8_iput(dst_uint8, src_uint8, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_uint16_iput(dst_uint16, src_uint16, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_uint32_iput(dst_uint32, src_uint32, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_uint64_iput(dst_uint64, src_uint64, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_size_iput(dst_size, src_size, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_ptrdiff_iput(dst_ptrdiff, src_ptrdiff, *dst_ptrdiff, *src_ptrdiff, count, target_pe);

  // nvshmemx_TYPENAME_iput_on_stream
  // ishmemx_TYPENAME_iput_on_queue
  // CHECK: ishmemx_iput_on_queue(dst_float, src_float, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iput_on_queue(dst_double, src_double, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iput_on_queue(dst_char, src_char, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iput_on_queue(dst_schar, src_schar, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iput_on_queue(dst_short, src_short, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iput_on_queue(dst_int, src_int, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iput_on_queue(dst_long, src_long, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iput_on_queue(dst_longlong, src_longlong, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iput_on_queue(dst_uchar, src_uchar, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iput_on_queue(dst_ushort, src_ushort, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iput_on_queue(dst_uint, src_uint, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iput_on_queue(dst_ulong, src_ulong, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iput_on_queue(dst_ulonglong, src_ulonglong, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iput_on_queue(dst_int8, src_int8, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iput_on_queue(dst_int16, src_int16, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iput_on_queue(dst_int32, src_int32, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iput_on_queue(dst_int64, src_int64, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iput_on_queue(dst_uint8, src_uint8, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iput_on_queue(dst_uint16, src_uint16, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iput_on_queue(dst_uint32, src_uint32, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iput_on_queue(dst_uint64, src_uint64, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iput_on_queue(dst_size, src_size, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iput_on_queue(dst_ptrdiff, src_ptrdiff, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  nvshmemx_float_iput_on_stream(dst_float, src_float, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_double_iput_on_stream(dst_double, src_double, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_char_iput_on_stream(dst_char, src_char, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_schar_iput_on_stream(dst_schar, src_schar, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_short_iput_on_stream(dst_short, src_short, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_int_iput_on_stream(dst_int, src_int, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_long_iput_on_stream(dst_long, src_long, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_longlong_iput_on_stream(dst_longlong, src_longlong, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_uchar_iput_on_stream(dst_uchar, src_uchar, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_ushort_iput_on_stream(dst_ushort, src_ushort, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_uint_iput_on_stream(dst_uint, src_uint, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_ulong_iput_on_stream(dst_ulong, src_ulong, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_ulonglong_iput_on_stream(dst_ulonglong, src_ulonglong, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_int8_iput_on_stream(dst_int8, src_int8, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_int16_iput_on_stream(dst_int16, src_int16, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_int32_iput_on_stream(dst_int32, src_int32, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_int64_iput_on_stream(dst_int64, src_int64, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_uint8_iput_on_stream(dst_uint8, src_uint8, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_uint16_iput_on_stream(dst_uint16, src_uint16, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_uint32_iput_on_stream(dst_uint32, src_uint32, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_uint64_iput_on_stream(dst_uint64, src_uint64, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_size_iput_on_stream(dst_size, src_size, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_ptrdiff_iput_on_stream(dst_ptrdiff, src_ptrdiff, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);

  // nvshmem_iputSIZE
  // ishmem_iputSIZE
  // CHECK: ishmem_iput8(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput16(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput32(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput64(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iput128(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_iput8(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_iput16(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_iput32(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_iput64(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_iput128(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);

  // nvshmemx_iputSIZE_on_stream
  // ishmemx_iputSIZE_on_queue
  // CHECK: ishmemx_iput8_on_queue(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iput16_on_queue(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iput32_on_queue(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iput64_on_queue(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iput128_on_queue(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  nvshmemx_iput8_on_stream(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_iput16_on_stream(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_iput32_on_stream(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_iput64_on_stream(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_iput128_on_stream(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);

  // nvshmem_putmem
  // ishmem_putmem
  // CHECK: ishmem_putmem(dst_void, src_void, count, target_pe);
  nvshmem_putmem(dst_void, src_void, count, target_pe);

  // nvshmemx_putmem_on_stream
  // ishmemx_putmem_on_queue
  // CHECK: ishmemx_putmem_on_queue(dst_void, src_void, count, target_pe, *stream);
  nvshmemx_putmem_on_stream(dst_void, src_void, count, target_pe, stream);

  // nvshmem_TYPENAME_p
  // ishmem_TYPENAME_p
  // CHECK: ishmem_p(dst_float, *src_float, target_pe);
  // CHECK-NEXT: ishmem_p(dst_double, *src_double, target_pe);
  // CHECK-NEXT: ishmem_p(dst_char, *src_char, target_pe);
  // CHECK-NEXT: ishmem_p(dst_schar, *src_schar, target_pe);
  // CHECK-NEXT: ishmem_p(dst_short, *src_short, target_pe);
  // CHECK-NEXT: ishmem_p(dst_int, *src_int, target_pe);
  // CHECK-NEXT: ishmem_p(dst_long, *src_long, target_pe);
  // CHECK-NEXT: ishmem_p(dst_longlong, *src_longlong, target_pe);
  // CHECK-NEXT: ishmem_p(dst_uchar, *src_uchar, target_pe);
  // CHECK-NEXT: ishmem_p(dst_ushort, *src_ushort, target_pe);
  // CHECK-NEXT: ishmem_p(dst_uint, *src_uint, target_pe);
  // CHECK-NEXT: ishmem_p(dst_ulong, *src_ulong, target_pe);
  // CHECK-NEXT: ishmem_p(dst_ulonglong, *src_ulonglong, target_pe);
  // CHECK-NEXT: ishmem_p(dst_int8, *src_int8, target_pe);
  // CHECK-NEXT: ishmem_p(dst_int16, *src_int16, target_pe);
  // CHECK-NEXT: ishmem_p(dst_int32, *src_int32, target_pe);
  // CHECK-NEXT: ishmem_p(dst_int64, *src_int64, target_pe);
  // CHECK-NEXT: ishmem_p(dst_uint8, *src_uint8, target_pe);
  // CHECK-NEXT: ishmem_p(dst_uint16, *src_uint16, target_pe);
  // CHECK-NEXT: ishmem_p(dst_uint32, *src_uint32, target_pe);
  // CHECK-NEXT: ishmem_p(dst_uint64, *src_uint64, target_pe);
  // CHECK-NEXT: ishmem_p(dst_size, *src_size, target_pe);
  // CHECK-NEXT: ishmem_p(dst_ptrdiff, *src_ptrdiff, target_pe);
  nvshmem_float_p(dst_float, *src_float, target_pe);
  nvshmem_double_p(dst_double, *src_double, target_pe);
  nvshmem_char_p(dst_char, *src_char, target_pe);
  nvshmem_schar_p(dst_schar, *src_schar, target_pe);
  nvshmem_short_p(dst_short, *src_short, target_pe);
  nvshmem_int_p(dst_int, *src_int, target_pe);
  nvshmem_long_p(dst_long, *src_long, target_pe);
  nvshmem_longlong_p(dst_longlong, *src_longlong, target_pe);
  nvshmem_uchar_p(dst_uchar, *src_uchar, target_pe);
  nvshmem_ushort_p(dst_ushort, *src_ushort, target_pe);
  nvshmem_uint_p(dst_uint, *src_uint, target_pe);
  nvshmem_ulong_p(dst_ulong, *src_ulong, target_pe);
  nvshmem_ulonglong_p(dst_ulonglong, *src_ulonglong, target_pe);
  nvshmem_int8_p(dst_int8, *src_int8, target_pe);
  nvshmem_int16_p(dst_int16, *src_int16, target_pe);
  nvshmem_int32_p(dst_int32, *src_int32, target_pe);
  nvshmem_int64_p(dst_int64, *src_int64, target_pe);
  nvshmem_uint8_p(dst_uint8, *src_uint8, target_pe);
  nvshmem_uint16_p(dst_uint16, *src_uint16, target_pe);
  nvshmem_uint32_p(dst_uint32, *src_uint32, target_pe);
  nvshmem_uint64_p(dst_uint64, *src_uint64, target_pe);
  nvshmem_size_p(dst_size, *src_size, target_pe);
  nvshmem_ptrdiff_p(dst_ptrdiff, *src_ptrdiff, target_pe);

  // nvshmem_TYPENAME_get
  // ishmem_TYPENAME_get
  // CHECK: ishmem_get(dst_float, src_float, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_double, src_double, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_char, src_char, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_schar, src_schar, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_short, src_short, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_int, src_int, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_long, src_long, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_longlong, src_longlong, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_uchar, src_uchar, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_ushort, src_ushort, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_uint, src_uint, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_ulong, src_ulong, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_ulonglong, src_ulonglong, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_int8, src_int8, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_int16, src_int16, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_int32, src_int32, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_int64, src_int64, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_uint8, src_uint8, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_uint16, src_uint16, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_uint32, src_uint32, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_uint64, src_uint64, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_size, src_size, count, target_pe);
  // CHECK-NEXT: ishmem_get(dst_ptrdiff, src_ptrdiff, count, target_pe);
  nvshmem_float_get(dst_float, src_float, count, target_pe);
  nvshmem_double_get(dst_double, src_double, count, target_pe);
  nvshmem_char_get(dst_char, src_char, count, target_pe);
  nvshmem_schar_get(dst_schar, src_schar, count, target_pe);
  nvshmem_short_get(dst_short, src_short, count, target_pe);
  nvshmem_int_get(dst_int, src_int, count, target_pe);
  nvshmem_long_get(dst_long, src_long, count, target_pe);
  nvshmem_longlong_get(dst_longlong, src_longlong, count, target_pe);
  nvshmem_uchar_get(dst_uchar, src_uchar, count, target_pe);
  nvshmem_ushort_get(dst_ushort, src_ushort, count, target_pe);
  nvshmem_uint_get(dst_uint, src_uint, count, target_pe);
  nvshmem_ulong_get(dst_ulong, src_ulong, count, target_pe);
  nvshmem_ulonglong_get(dst_ulonglong, src_ulonglong, count, target_pe);
  nvshmem_int8_get(dst_int8, src_int8, count, target_pe);
  nvshmem_int16_get(dst_int16, src_int16, count, target_pe);
  nvshmem_int32_get(dst_int32, src_int32, count, target_pe);
  nvshmem_int64_get(dst_int64, src_int64, count, target_pe);
  nvshmem_uint8_get(dst_uint8, src_uint8, count, target_pe);
  nvshmem_uint16_get(dst_uint16, src_uint16, count, target_pe);
  nvshmem_uint32_get(dst_uint32, src_uint32, count, target_pe);
  nvshmem_uint64_get(dst_uint64, src_uint64, count, target_pe);
  nvshmem_size_get(dst_size, src_size, count, target_pe);
  nvshmem_ptrdiff_get(dst_ptrdiff, src_ptrdiff, count, target_pe);

  // nvshmemx_TYPENAME_get_on_stream
  // ishmemx_TYPENAME_get_on_queue
  // CHECK: ishmemx_get_on_queue(dst_float, src_float, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_on_queue(dst_double, src_double, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_on_queue(dst_char, src_char, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_on_queue(dst_schar, src_schar, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_on_queue(dst_short, src_short, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_on_queue(dst_int, src_int, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_on_queue(dst_long, src_long, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_on_queue(dst_longlong, src_longlong, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_on_queue(dst_uchar, src_uchar, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_on_queue(dst_ushort, src_ushort, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_on_queue(dst_uint, src_uint, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_on_queue(dst_ulong, src_ulong, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_on_queue(dst_ulonglong, src_ulonglong, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_on_queue(dst_int8, src_int8, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_on_queue(dst_int16, src_int16, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_on_queue(dst_int32, src_int32, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_on_queue(dst_int64, src_int64, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_on_queue(dst_uint8, src_uint8, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_on_queue(dst_uint16, src_uint16, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_on_queue(dst_uint32, src_uint32, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_on_queue(dst_uint64, src_uint64, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_on_queue(dst_size, src_size, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_on_queue(dst_ptrdiff, src_ptrdiff, count, target_pe, *stream);
  nvshmemx_float_get_on_stream(dst_float, src_float, count, target_pe, stream);
  nvshmemx_double_get_on_stream(dst_double, src_double, count, target_pe, stream);
  nvshmemx_char_get_on_stream(dst_char, src_char, count, target_pe, stream);
  nvshmemx_schar_get_on_stream(dst_schar, src_schar, count, target_pe, stream);
  nvshmemx_short_get_on_stream(dst_short, src_short, count, target_pe, stream);
  nvshmemx_int_get_on_stream(dst_int, src_int, count, target_pe, stream);
  nvshmemx_long_get_on_stream(dst_long, src_long, count, target_pe, stream);
  nvshmemx_longlong_get_on_stream(dst_longlong, src_longlong, count, target_pe, stream);
  nvshmemx_uchar_get_on_stream(dst_uchar, src_uchar, count, target_pe, stream);
  nvshmemx_ushort_get_on_stream(dst_ushort, src_ushort, count, target_pe, stream);
  nvshmemx_uint_get_on_stream(dst_uint, src_uint, count, target_pe, stream);
  nvshmemx_ulong_get_on_stream(dst_ulong, src_ulong, count, target_pe, stream);
  nvshmemx_ulonglong_get_on_stream(dst_ulonglong, src_ulonglong, count, target_pe, stream);
  nvshmemx_int8_get_on_stream(dst_int8, src_int8, count, target_pe, stream);
  nvshmemx_int16_get_on_stream(dst_int16, src_int16, count, target_pe, stream);
  nvshmemx_int32_get_on_stream(dst_int32, src_int32, count, target_pe, stream);
  nvshmemx_int64_get_on_stream(dst_int64, src_int64, count, target_pe, stream);
  nvshmemx_uint8_get_on_stream(dst_uint8, src_uint8, count, target_pe, stream);
  nvshmemx_uint16_get_on_stream(dst_uint16, src_uint16, count, target_pe, stream);
  nvshmemx_uint32_get_on_stream(dst_uint32, src_uint32, count, target_pe, stream);
  nvshmemx_uint64_get_on_stream(dst_uint64, src_uint64, count, target_pe, stream);
  nvshmemx_size_get_on_stream(dst_size, src_size, count, target_pe, stream);
  nvshmemx_ptrdiff_get_on_stream(dst_ptrdiff, src_ptrdiff, count, target_pe, stream);

  // nvshmem_getSIZE
  // ishmem_getSIZE
  // CHECK: ishmem_get8(dst_void, src_void, count, target_pe);
  // CHECK-NEXT: ishmem_get16(dst_void, src_void, count, target_pe);
  // CHECK-NEXT: ishmem_get32(dst_void, src_void, count, target_pe);
  // CHECK-NEXT: ishmem_get64(dst_void, src_void, count, target_pe);
  // CHECK-NEXT: ishmem_get128(dst_void, src_void, count, target_pe);
  nvshmem_get8(dst_void, src_void, count, target_pe);
  nvshmem_get16(dst_void, src_void, count, target_pe);
  nvshmem_get32(dst_void, src_void, count, target_pe);
  nvshmem_get64(dst_void, src_void, count, target_pe);
  nvshmem_get128(dst_void, src_void, count, target_pe);

  // nvshmemx_getSIZE_on_stream
  // ishmemx_getSIZE_on_queue
  // CHECK: ishmemx_get8_on_queue(dst_void, src_void, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get16_on_queue(dst_void, src_void, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get32_on_queue(dst_void, src_void, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get64_on_queue(dst_void, src_void, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get128_on_queue(dst_void, src_void, count, target_pe, *stream);
  nvshmemx_get8_on_stream(dst_void, src_void, count, target_pe, stream);
  nvshmemx_get16_on_stream(dst_void, src_void, count, target_pe, stream);
  nvshmemx_get32_on_stream(dst_void, src_void, count, target_pe, stream);
  nvshmemx_get64_on_stream(dst_void, src_void, count, target_pe, stream);
  nvshmemx_get128_on_stream(dst_void, src_void, count, target_pe, stream);

  // nvshmem_TYPENAME_iget
  // ishmem_TYPENAME_iget
  // CHECK: ishmem_iget(dst_float, src_float, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_double, src_double, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_char, src_char, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_schar, src_schar, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_short, src_short, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_int, src_int, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_long, src_long, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_longlong, src_longlong, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_uchar, src_uchar, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_ushort, src_ushort, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_uint, src_uint, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_ulong, src_ulong, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_ulonglong, src_ulonglong, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_int8, src_int8, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_int16, src_int16, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_int32, src_int32, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_int64, src_int64, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_uint8, src_uint8, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_uint16, src_uint16, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_uint32, src_uint32, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_uint64, src_uint64, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_size, src_size, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget(dst_ptrdiff, src_ptrdiff, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_float_iget(dst_float, src_float, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_double_iget(dst_double, src_double, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_char_iget(dst_char, src_char, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_schar_iget(dst_schar, src_schar, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_short_iget(dst_short, src_short, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_int_iget(dst_int, src_int, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_long_iget(dst_long, src_long, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_longlong_iget(dst_longlong, src_longlong, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_uchar_iget(dst_uchar, src_uchar, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_ushort_iget(dst_ushort, src_ushort, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_uint_iget(dst_uint, src_uint, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_ulong_iget(dst_ulong, src_ulong, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_ulonglong_iget(dst_ulonglong, src_ulonglong, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_int8_iget(dst_int8, src_int8, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_int16_iget(dst_int16, src_int16, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_int32_iget(dst_int32, src_int32, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_int64_iget(dst_int64, src_int64, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_uint8_iget(dst_uint8, src_uint8, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_uint16_iget(dst_uint16, src_uint16, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_uint32_iget(dst_uint32, src_uint32, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_uint64_iget(dst_uint64, src_uint64, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_size_iget(dst_size, src_size, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_ptrdiff_iget(dst_ptrdiff, src_ptrdiff, *dst_ptrdiff, *src_ptrdiff, count, target_pe);

  // nvshmemx_TYPENAME_iget_on_stream
  // ishmemx_TYPENAME_iget_on_queue
  // CHECK: ishmemx_iget_on_queue(dst_float, src_float, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iget_on_queue(dst_double, src_double, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iget_on_queue(dst_char, src_char, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iget_on_queue(dst_schar, src_schar, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iget_on_queue(dst_short, src_short, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iget_on_queue(dst_int, src_int, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iget_on_queue(dst_long, src_long, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iget_on_queue(dst_longlong, src_longlong, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iget_on_queue(dst_uchar, src_uchar, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iget_on_queue(dst_ushort, src_ushort, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iget_on_queue(dst_uint, src_uint, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iget_on_queue(dst_ulong, src_ulong, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iget_on_queue(dst_ulonglong, src_ulonglong, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iget_on_queue(dst_int8, src_int8, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iget_on_queue(dst_int16, src_int16, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iget_on_queue(dst_int32, src_int32, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iget_on_queue(dst_int64, src_int64, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iget_on_queue(dst_uint8, src_uint8, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iget_on_queue(dst_uint16, src_uint16, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iget_on_queue(dst_uint32, src_uint32, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iget_on_queue(dst_uint64, src_uint64, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iget_on_queue(dst_size, src_size, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iget_on_queue(dst_ptrdiff, src_ptrdiff, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  nvshmemx_float_iget_on_stream(dst_float, src_float, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_double_iget_on_stream(dst_double, src_double, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_char_iget_on_stream(dst_char, src_char, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_schar_iget_on_stream(dst_schar, src_schar, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_short_iget_on_stream(dst_short, src_short, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_int_iget_on_stream(dst_int, src_int, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_long_iget_on_stream(dst_long, src_long, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_longlong_iget_on_stream(dst_longlong, src_longlong, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_uchar_iget_on_stream(dst_uchar, src_uchar, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_ushort_iget_on_stream(dst_ushort, src_ushort, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_uint_iget_on_stream(dst_uint, src_uint, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_ulong_iget_on_stream(dst_ulong, src_ulong, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_ulonglong_iget_on_stream(dst_ulonglong, src_ulonglong, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_int8_iget_on_stream(dst_int8, src_int8, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_int16_iget_on_stream(dst_int16, src_int16, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_int32_iget_on_stream(dst_int32, src_int32, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_int64_iget_on_stream(dst_int64, src_int64, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_uint8_iget_on_stream(dst_uint8, src_uint8, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_uint16_iget_on_stream(dst_uint16, src_uint16, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_uint32_iget_on_stream(dst_uint32, src_uint32, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_uint64_iget_on_stream(dst_uint64, src_uint64, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_size_iget_on_stream(dst_size, src_size, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_ptrdiff_iget_on_stream(dst_ptrdiff, src_ptrdiff, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);

  // nvshmem_igetSIZE
  // ishmem_igetSIZE
  // CHECK: ishmem_iget8(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget16(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget32(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget64(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  // CHECK-NEXT: ishmem_iget128(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_iget8(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_iget16(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_iget32(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_iget64(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);
  nvshmem_iget128(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe);

  // nvshmemx_igetSIZE_on_stream
  // ishmemx_igetSIZE_on_queue
  // CHECK: ishmemx_iget8_on_queue(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iget16_on_queue(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iget32_on_queue(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iget64_on_queue(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_iget128_on_queue(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, *stream);
  nvshmemx_iget8_on_stream(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_iget16_on_stream(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_iget32_on_stream(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_iget64_on_stream(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);
  nvshmemx_iget128_on_stream(dst_void, src_void, *dst_ptrdiff, *src_ptrdiff, count, target_pe, stream);

  // nvshmem_getmem
  // ishmem_getmem
  // CHECK: ishmem_getmem(dst_void, src_void, count, target_pe);
  nvshmem_getmem(dst_void, src_void, count, target_pe);

  // nvshmemx_getmem_on_stream
  // ishmemx_getmem_on_queue
  // CHECK: ishmemx_getmem_on_queue(dst_void, src_void, count, target_pe, *stream);
  nvshmemx_getmem_on_stream(dst_void, src_void, count, target_pe, stream);

  // nvshmem_TYPENAME_g
  // ishmem_TYPENAME_g
  // CHECK: *dst_float = ishmem_g(src_float, target_pe);
  // CHECK-NEXT: *dst_double = ishmem_g(src_double, target_pe);
  // CHECK-NEXT: *dst_char = ishmem_g(src_char, target_pe);
  // CHECK-NEXT: *dst_schar = ishmem_g(src_schar, target_pe);
  // CHECK-NEXT: *dst_short = ishmem_g(src_short, target_pe);
  // CHECK-NEXT: *dst_int = ishmem_g(src_int, target_pe);
  // CHECK-NEXT: *dst_long = ishmem_g(src_long, target_pe);
  // CHECK-NEXT: *dst_longlong = ishmem_g(src_longlong, target_pe);
  // CHECK-NEXT: *dst_uchar = ishmem_g(src_uchar, target_pe);
  // CHECK-NEXT: *dst_ushort = ishmem_g(src_ushort, target_pe);
  // CHECK-NEXT: *dst_uint = ishmem_g(src_uint, target_pe);
  // CHECK-NEXT: *dst_ulong = ishmem_g(src_ulong, target_pe);
  // CHECK-NEXT: *dst_ulonglong = ishmem_g(src_ulonglong, target_pe);
  // CHECK-NEXT: *dst_int8 = ishmem_g(src_int8, target_pe);
  // CHECK-NEXT: *dst_int16 = ishmem_g(src_int16, target_pe);
  // CHECK-NEXT: *dst_int32 = ishmem_g(src_int32, target_pe);
  // CHECK-NEXT: *dst_int64 = ishmem_g(src_int64, target_pe);
  // CHECK-NEXT: *dst_uint8 = ishmem_g(src_uint8, target_pe);
  // CHECK-NEXT: *dst_uint16 = ishmem_g(src_uint16, target_pe);
  // CHECK-NEXT: *dst_uint32 = ishmem_g(src_uint32, target_pe);
  // CHECK-NEXT: *dst_uint64 = ishmem_g(src_uint64, target_pe);
  // CHECK-NEXT: *dst_size = ishmem_g(src_size, target_pe);
  // CHECK-NEXT: *dst_ptrdiff = ishmem_g(src_ptrdiff, target_pe);
  *dst_float = nvshmem_float_g(src_float, target_pe);
  *dst_double = nvshmem_double_g(src_double, target_pe);
  *dst_char = nvshmem_char_g(src_char, target_pe);
  *dst_schar = nvshmem_schar_g(src_schar, target_pe);
  *dst_short = nvshmem_short_g(src_short, target_pe);
  *dst_int = nvshmem_int_g(src_int, target_pe);
  *dst_long = nvshmem_long_g(src_long, target_pe);
  *dst_longlong = nvshmem_longlong_g(src_longlong, target_pe);
  *dst_uchar = nvshmem_uchar_g(src_uchar, target_pe);
  *dst_ushort = nvshmem_ushort_g(src_ushort, target_pe);
  *dst_uint = nvshmem_uint_g(src_uint, target_pe);
  *dst_ulong = nvshmem_ulong_g(src_ulong, target_pe);
  *dst_ulonglong = nvshmem_ulonglong_g(src_ulonglong, target_pe);
  *dst_int8 = nvshmem_int8_g(src_int8, target_pe);
  *dst_int16 = nvshmem_int16_g(src_int16, target_pe);
  *dst_int32 = nvshmem_int32_g(src_int32, target_pe);
  *dst_int64 = nvshmem_int64_g(src_int64, target_pe);
  *dst_uint8 = nvshmem_uint8_g(src_uint8, target_pe);
  *dst_uint16 = nvshmem_uint16_g(src_uint16, target_pe);
  *dst_uint32 = nvshmem_uint32_g(src_uint32, target_pe);
  *dst_uint64 = nvshmem_uint64_g(src_uint64, target_pe);
  *dst_size = nvshmem_size_g(src_size, target_pe);
  *dst_ptrdiff = nvshmem_ptrdiff_g(src_ptrdiff, target_pe);

  return 0;
}
