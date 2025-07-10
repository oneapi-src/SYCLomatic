// REQUIRES: system-linux
// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7
// RUN: dpct --format-range=none -out-root %T/nvshmem %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/nvshmem/rma_nbi.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST %T/nvshmem/rma_nbi.dp.cpp -o %T/nvshmem/rma_nbi.dp.o %}
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

  // nvshmem_TYPENAME_put_nbi
  // ishmem_TYPENAME_put_nbi
  // CHECK: ishmem_put_nbi(dst_float, src_float, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_double, src_double, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_char, src_char, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_schar, src_schar, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_short, src_short, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_int, src_int, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_long, src_long, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_longlong, src_longlong, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_uchar, src_uchar, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_ushort, src_ushort, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_uint, src_uint, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_ulong, src_ulong, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_ulonglong, src_ulonglong, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_int8, src_int8, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_int16, src_int16, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_int32, src_int32, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_int64, src_int64, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_uint8, src_uint8, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_uint16, src_uint16, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_uint32, src_uint32, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_uint64, src_uint64, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_size, src_size, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_ptrdiff, src_ptrdiff, count, target_pe);
  nvshmem_float_put_nbi(dst_float, src_float, count, target_pe);
  nvshmem_double_put_nbi(dst_double, src_double, count, target_pe);
  nvshmem_char_put_nbi(dst_char, src_char, count, target_pe);
  nvshmem_schar_put_nbi(dst_schar, src_schar, count, target_pe);
  nvshmem_short_put_nbi(dst_short, src_short, count, target_pe);
  nvshmem_int_put_nbi(dst_int, src_int, count, target_pe);
  nvshmem_long_put_nbi(dst_long, src_long, count, target_pe);
  nvshmem_longlong_put_nbi(dst_longlong, src_longlong, count, target_pe);
  nvshmem_uchar_put_nbi(dst_uchar, src_uchar, count, target_pe);
  nvshmem_ushort_put_nbi(dst_ushort, src_ushort, count, target_pe);
  nvshmem_uint_put_nbi(dst_uint, src_uint, count, target_pe);
  nvshmem_ulong_put_nbi(dst_ulong, src_ulong, count, target_pe);
  nvshmem_ulonglong_put_nbi(dst_ulonglong, src_ulonglong, count, target_pe);
  nvshmem_int8_put_nbi(dst_int8, src_int8, count, target_pe);
  nvshmem_int16_put_nbi(dst_int16, src_int16, count, target_pe);
  nvshmem_int32_put_nbi(dst_int32, src_int32, count, target_pe);
  nvshmem_int64_put_nbi(dst_int64, src_int64, count, target_pe);
  nvshmem_uint8_put_nbi(dst_uint8, src_uint8, count, target_pe);
  nvshmem_uint16_put_nbi(dst_uint16, src_uint16, count, target_pe);
  nvshmem_uint32_put_nbi(dst_uint32, src_uint32, count, target_pe);
  nvshmem_uint64_put_nbi(dst_uint64, src_uint64, count, target_pe);
  nvshmem_size_put_nbi(dst_size, src_size, count, target_pe);
  nvshmem_ptrdiff_put_nbi(dst_ptrdiff, src_ptrdiff, count, target_pe);

  // nvshmemx_TYPENAME_put_nbi_block
  // ishmemx_TYPENAME_put_nbi_work_group
  // CHECK: ishmemx_put_nbi_work_group(dst_float, src_float, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_double, src_double, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_char, src_char, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_schar, src_schar, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_short, src_short, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_int, src_int, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_long, src_long, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_longlong, src_longlong, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_uchar, src_uchar, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_ushort, src_ushort, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_uint, src_uint, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_ulong, src_ulong, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_ulonglong, src_ulonglong, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_int8, src_int8, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_int16, src_int16, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_int32, src_int32, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_int64, src_int64, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_uint8, src_uint8, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_uint16, src_uint16, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_uint32, src_uint32, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_uint64, src_uint64, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_size, src_size, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_ptrdiff, src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  nvshmemx_float_put_nbi_block(dst_float, src_float, count, target_pe);
  nvshmemx_double_put_nbi_block(dst_double, src_double, count, target_pe);
  nvshmemx_char_put_nbi_block(dst_char, src_char, count, target_pe);
  nvshmemx_schar_put_nbi_block(dst_schar, src_schar, count, target_pe);
  nvshmemx_short_put_nbi_block(dst_short, src_short, count, target_pe);
  nvshmemx_int_put_nbi_block(dst_int, src_int, count, target_pe);
  nvshmemx_long_put_nbi_block(dst_long, src_long, count, target_pe);
  nvshmemx_longlong_put_nbi_block(dst_longlong, src_longlong, count, target_pe);
  nvshmemx_uchar_put_nbi_block(dst_uchar, src_uchar, count, target_pe);
  nvshmemx_ushort_put_nbi_block(dst_ushort, src_ushort, count, target_pe);
  nvshmemx_uint_put_nbi_block(dst_uint, src_uint, count, target_pe);
  nvshmemx_ulong_put_nbi_block(dst_ulong, src_ulong, count, target_pe);
  nvshmemx_ulonglong_put_nbi_block(dst_ulonglong, src_ulonglong, count, target_pe);
  nvshmemx_int8_put_nbi_block(dst_int8, src_int8, count, target_pe);
  nvshmemx_int16_put_nbi_block(dst_int16, src_int16, count, target_pe);
  nvshmemx_int32_put_nbi_block(dst_int32, src_int32, count, target_pe);
  nvshmemx_int64_put_nbi_block(dst_int64, src_int64, count, target_pe);
  nvshmemx_uint8_put_nbi_block(dst_uint8, src_uint8, count, target_pe);
  nvshmemx_uint16_put_nbi_block(dst_uint16, src_uint16, count, target_pe);
  nvshmemx_uint32_put_nbi_block(dst_uint32, src_uint32, count, target_pe);
  nvshmemx_uint64_put_nbi_block(dst_uint64, src_uint64, count, target_pe);
  nvshmemx_size_put_nbi_block(dst_size, src_size, count, target_pe);
  nvshmemx_ptrdiff_put_nbi_block(dst_ptrdiff, src_ptrdiff, count, target_pe);

  // nvshmemx_TYPENAME_put_nbi_warp
  // ishmemx_TYPENAME_put_nbi_work_group
  // CHECK: ishmemx_put_nbi_work_group(dst_float, src_float, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_double, src_double, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_char, src_char, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_schar, src_schar, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_short, src_short, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_int, src_int, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_long, src_long, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_longlong, src_longlong, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_uchar, src_uchar, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_ushort, src_ushort, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_uint, src_uint, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_ulong, src_ulong, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_ulonglong, src_ulonglong, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_int8, src_int8, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_int16, src_int16, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_int32, src_int32, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_int64, src_int64, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_uint8, src_uint8, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_uint16, src_uint16, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_uint32, src_uint32, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_uint64, src_uint64, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_size, src_size, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put_nbi_work_group(dst_ptrdiff, src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  nvshmemx_float_put_nbi_warp(dst_float, src_float, count, target_pe);
  nvshmemx_double_put_nbi_warp(dst_double, src_double, count, target_pe);
  nvshmemx_char_put_nbi_warp(dst_char, src_char, count, target_pe);
  nvshmemx_schar_put_nbi_warp(dst_schar, src_schar, count, target_pe);
  nvshmemx_short_put_nbi_warp(dst_short, src_short, count, target_pe);
  nvshmemx_int_put_nbi_warp(dst_int, src_int, count, target_pe);
  nvshmemx_long_put_nbi_warp(dst_long, src_long, count, target_pe);
  nvshmemx_longlong_put_nbi_warp(dst_longlong, src_longlong, count, target_pe);
  nvshmemx_uchar_put_nbi_warp(dst_uchar, src_uchar, count, target_pe);
  nvshmemx_ushort_put_nbi_warp(dst_ushort, src_ushort, count, target_pe);
  nvshmemx_uint_put_nbi_warp(dst_uint, src_uint, count, target_pe);
  nvshmemx_ulong_put_nbi_warp(dst_ulong, src_ulong, count, target_pe);
  nvshmemx_ulonglong_put_nbi_warp(dst_ulonglong, src_ulonglong, count, target_pe);
  nvshmemx_int8_put_nbi_warp(dst_int8, src_int8, count, target_pe);
  nvshmemx_int16_put_nbi_warp(dst_int16, src_int16, count, target_pe);
  nvshmemx_int32_put_nbi_warp(dst_int32, src_int32, count, target_pe);
  nvshmemx_int64_put_nbi_warp(dst_int64, src_int64, count, target_pe);
  nvshmemx_uint8_put_nbi_warp(dst_uint8, src_uint8, count, target_pe);
  nvshmemx_uint16_put_nbi_warp(dst_uint16, src_uint16, count, target_pe);
  nvshmemx_uint32_put_nbi_warp(dst_uint32, src_uint32, count, target_pe);
  nvshmemx_uint64_put_nbi_warp(dst_uint64, src_uint64, count, target_pe);
  nvshmemx_size_put_nbi_warp(dst_size, src_size, count, target_pe);
  nvshmemx_ptrdiff_put_nbi_warp(dst_ptrdiff, src_ptrdiff, count, target_pe);

  // nvshmem_putSIZE_nbi
  // ishmem_putSIZE_nbi
  // CHECK: ishmem_put8_nbi(dst_void, src_void, count, target_pe);
  // CHECK-NEXT: ishmem_put16_nbi(dst_void, src_void, count, target_pe);
  // CHECK-NEXT: ishmem_put32_nbi(dst_void, src_void, count, target_pe);
  // CHECK-NEXT: ishmem_put64_nbi(dst_void, src_void, count, target_pe);
  // CHECK-NEXT: ishmem_put128_nbi(dst_void, src_void, count, target_pe);
  nvshmem_put8_nbi(dst_void, src_void, count, target_pe);
  nvshmem_put16_nbi(dst_void, src_void, count, target_pe);
  nvshmem_put32_nbi(dst_void, src_void, count, target_pe);
  nvshmem_put64_nbi(dst_void, src_void, count, target_pe);
  nvshmem_put128_nbi(dst_void, src_void, count, target_pe);

  // nvshmemx_putSIZE_nbi_block
  // ishmemx_putSIZE_nbi_work_group
  // CHECK: ishmemx_put8_nbi_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put16_nbi_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put32_nbi_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put64_nbi_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_put128_nbi_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  nvshmemx_put8_nbi_block(dst_void, src_void, count, target_pe);
  nvshmemx_put16_nbi_block(dst_void, src_void, count, target_pe);
  nvshmemx_put32_nbi_block(dst_void, src_void, count, target_pe);
  nvshmemx_put64_nbi_block(dst_void, src_void, count, target_pe);
  nvshmemx_put128_nbi_block(dst_void, src_void, count, target_pe);

  // nvshmemx_putSIZE_nbi_warp
  // ishmemx_putSIZE_nbi_work_group
  // CHECK: ishmemx_put8_nbi_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put16_nbi_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put32_nbi_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put64_nbi_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_put128_nbi_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  nvshmemx_put8_nbi_warp(dst_void, src_void, count, target_pe);
  nvshmemx_put16_nbi_warp(dst_void, src_void, count, target_pe);
  nvshmemx_put32_nbi_warp(dst_void, src_void, count, target_pe);
  nvshmemx_put64_nbi_warp(dst_void, src_void, count, target_pe);
  nvshmemx_put128_nbi_warp(dst_void, src_void, count, target_pe);

  // nvshmem_putmem_nbi
  // ishmem_putmem_nbi
  // CHECK: ishmem_putmem_nbi(dst_void, src_void, count, target_pe);
  nvshmem_putmem_nbi(dst_void, src_void, count, target_pe);

  // nvshmemx_putmem_nbi_block
  // ishmemx_putmem_nbi_work_group
  // CHECK: ishmemx_putmem_nbi_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  nvshmemx_putmem_nbi_block(dst_void, src_void, count, target_pe);

  // nvshmemx_putmem_nbi_warp
  // ishmemx_putmem_nbi_work_group
  // CHECK: ishmemx_putmem_nbi_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  nvshmemx_putmem_nbi_warp(dst_void, src_void, count, target_pe);

  // nvshmem_TYPENAME_get_nbi
  // ishmem_TYPENAME_get_nbi
  // CHECK: ishmem_get_nbi(dst_float, src_float, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_double, src_double, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_char, src_char, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_schar, src_schar, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_short, src_short, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_int, src_int, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_long, src_long, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_longlong, src_longlong, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_uchar, src_uchar, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_ushort, src_ushort, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_uint, src_uint, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_ulong, src_ulong, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_ulonglong, src_ulonglong, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_int8, src_int8, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_int16, src_int16, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_int32, src_int32, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_int64, src_int64, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_uint8, src_uint8, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_uint16, src_uint16, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_uint32, src_uint32, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_uint64, src_uint64, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_size, src_size, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_ptrdiff, src_ptrdiff, count, target_pe);
  nvshmem_float_get_nbi(dst_float, src_float, count, target_pe);
  nvshmem_double_get_nbi(dst_double, src_double, count, target_pe);
  nvshmem_char_get_nbi(dst_char, src_char, count, target_pe);
  nvshmem_schar_get_nbi(dst_schar, src_schar, count, target_pe);
  nvshmem_short_get_nbi(dst_short, src_short, count, target_pe);
  nvshmem_int_get_nbi(dst_int, src_int, count, target_pe);
  nvshmem_long_get_nbi(dst_long, src_long, count, target_pe);
  nvshmem_longlong_get_nbi(dst_longlong, src_longlong, count, target_pe);
  nvshmem_uchar_get_nbi(dst_uchar, src_uchar, count, target_pe);
  nvshmem_ushort_get_nbi(dst_ushort, src_ushort, count, target_pe);
  nvshmem_uint_get_nbi(dst_uint, src_uint, count, target_pe);
  nvshmem_ulong_get_nbi(dst_ulong, src_ulong, count, target_pe);
  nvshmem_ulonglong_get_nbi(dst_ulonglong, src_ulonglong, count, target_pe);
  nvshmem_int8_get_nbi(dst_int8, src_int8, count, target_pe);
  nvshmem_int16_get_nbi(dst_int16, src_int16, count, target_pe);
  nvshmem_int32_get_nbi(dst_int32, src_int32, count, target_pe);
  nvshmem_int64_get_nbi(dst_int64, src_int64, count, target_pe);
  nvshmem_uint8_get_nbi(dst_uint8, src_uint8, count, target_pe);
  nvshmem_uint16_get_nbi(dst_uint16, src_uint16, count, target_pe);
  nvshmem_uint32_get_nbi(dst_uint32, src_uint32, count, target_pe);
  nvshmem_uint64_get_nbi(dst_uint64, src_uint64, count, target_pe);
  nvshmem_size_get_nbi(dst_size, src_size, count, target_pe);
  nvshmem_ptrdiff_get_nbi(dst_ptrdiff, src_ptrdiff, count, target_pe);

  // nvshmemx_TYPENAME_get_nbi_block
  // ishmemx_TYPENAME_get_nbi_work_group
  // CHECK: ishmemx_get_nbi_work_group(dst_float, src_float, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_double, src_double, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_char, src_char, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_schar, src_schar, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_short, src_short, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_int, src_int, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_long, src_long, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_longlong, src_longlong, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_uchar, src_uchar, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_ushort, src_ushort, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_uint, src_uint, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_ulong, src_ulong, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_ulonglong, src_ulonglong, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_int8, src_int8, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_int16, src_int16, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_int32, src_int32, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_int64, src_int64, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_uint8, src_uint8, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_uint16, src_uint16, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_uint32, src_uint32, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_uint64, src_uint64, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_size, src_size, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_ptrdiff, src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  nvshmemx_float_get_nbi_block(dst_float, src_float, count, target_pe);
  nvshmemx_double_get_nbi_block(dst_double, src_double, count, target_pe);
  nvshmemx_char_get_nbi_block(dst_char, src_char, count, target_pe);
  nvshmemx_schar_get_nbi_block(dst_schar, src_schar, count, target_pe);
  nvshmemx_short_get_nbi_block(dst_short, src_short, count, target_pe);
  nvshmemx_int_get_nbi_block(dst_int, src_int, count, target_pe);
  nvshmemx_long_get_nbi_block(dst_long, src_long, count, target_pe);
  nvshmemx_longlong_get_nbi_block(dst_longlong, src_longlong, count, target_pe);
  nvshmemx_uchar_get_nbi_block(dst_uchar, src_uchar, count, target_pe);
  nvshmemx_ushort_get_nbi_block(dst_ushort, src_ushort, count, target_pe);
  nvshmemx_uint_get_nbi_block(dst_uint, src_uint, count, target_pe);
  nvshmemx_ulong_get_nbi_block(dst_ulong, src_ulong, count, target_pe);
  nvshmemx_ulonglong_get_nbi_block(dst_ulonglong, src_ulonglong, count, target_pe);
  nvshmemx_int8_get_nbi_block(dst_int8, src_int8, count, target_pe);
  nvshmemx_int16_get_nbi_block(dst_int16, src_int16, count, target_pe);
  nvshmemx_int32_get_nbi_block(dst_int32, src_int32, count, target_pe);
  nvshmemx_int64_get_nbi_block(dst_int64, src_int64, count, target_pe);
  nvshmemx_uint8_get_nbi_block(dst_uint8, src_uint8, count, target_pe);
  nvshmemx_uint16_get_nbi_block(dst_uint16, src_uint16, count, target_pe);
  nvshmemx_uint32_get_nbi_block(dst_uint32, src_uint32, count, target_pe);
  nvshmemx_uint64_get_nbi_block(dst_uint64, src_uint64, count, target_pe);
  nvshmemx_size_get_nbi_block(dst_size, src_size, count, target_pe);
  nvshmemx_ptrdiff_get_nbi_block(dst_ptrdiff, src_ptrdiff, count, target_pe);

  // nvshmemx_TYPENAME_get_nbi_warp
  // ishmemx_TYPENAME_get_nbi_work_group
  // CHECK: ishmemx_get_nbi_work_group(dst_float, src_float, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_double, src_double, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_char, src_char, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_schar, src_schar, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_short, src_short, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_int, src_int, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_long, src_long, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_longlong, src_longlong, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_uchar, src_uchar, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_ushort, src_ushort, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_uint, src_uint, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_ulong, src_ulong, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_ulonglong, src_ulonglong, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_int8, src_int8, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_int16, src_int16, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_int32, src_int32, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_int64, src_int64, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_uint8, src_uint8, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_uint16, src_uint16, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_uint32, src_uint32, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_uint64, src_uint64, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_size, src_size, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get_nbi_work_group(dst_ptrdiff, src_ptrdiff, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  nvshmemx_float_get_nbi_warp(dst_float, src_float, count, target_pe);
  nvshmemx_double_get_nbi_warp(dst_double, src_double, count, target_pe);
  nvshmemx_char_get_nbi_warp(dst_char, src_char, count, target_pe);
  nvshmemx_schar_get_nbi_warp(dst_schar, src_schar, count, target_pe);
  nvshmemx_short_get_nbi_warp(dst_short, src_short, count, target_pe);
  nvshmemx_int_get_nbi_warp(dst_int, src_int, count, target_pe);
  nvshmemx_long_get_nbi_warp(dst_long, src_long, count, target_pe);
  nvshmemx_longlong_get_nbi_warp(dst_longlong, src_longlong, count, target_pe);
  nvshmemx_uchar_get_nbi_warp(dst_uchar, src_uchar, count, target_pe);
  nvshmemx_ushort_get_nbi_warp(dst_ushort, src_ushort, count, target_pe);
  nvshmemx_uint_get_nbi_warp(dst_uint, src_uint, count, target_pe);
  nvshmemx_ulong_get_nbi_warp(dst_ulong, src_ulong, count, target_pe);
  nvshmemx_ulonglong_get_nbi_warp(dst_ulonglong, src_ulonglong, count, target_pe);
  nvshmemx_int8_get_nbi_warp(dst_int8, src_int8, count, target_pe);
  nvshmemx_int16_get_nbi_warp(dst_int16, src_int16, count, target_pe);
  nvshmemx_int32_get_nbi_warp(dst_int32, src_int32, count, target_pe);
  nvshmemx_int64_get_nbi_warp(dst_int64, src_int64, count, target_pe);
  nvshmemx_uint8_get_nbi_warp(dst_uint8, src_uint8, count, target_pe);
  nvshmemx_uint16_get_nbi_warp(dst_uint16, src_uint16, count, target_pe);
  nvshmemx_uint32_get_nbi_warp(dst_uint32, src_uint32, count, target_pe);
  nvshmemx_uint64_get_nbi_warp(dst_uint64, src_uint64, count, target_pe);
  nvshmemx_size_get_nbi_warp(dst_size, src_size, count, target_pe);
  nvshmemx_ptrdiff_get_nbi_warp(dst_ptrdiff, src_ptrdiff, count, target_pe);

  // nvshmem_getSIZE_nbi
  // ishmem_getSIZE_nbi
  // CHECK: ishmem_get8_nbi(dst_void, src_void, count, target_pe);
  // CHECK-NEXT: ishmem_get16_nbi(dst_void, src_void, count, target_pe);
  // CHECK-NEXT: ishmem_get32_nbi(dst_void, src_void, count, target_pe);
  // CHECK-NEXT: ishmem_get64_nbi(dst_void, src_void, count, target_pe);
  // CHECK-NEXT: ishmem_get128_nbi(dst_void, src_void, count, target_pe);
  nvshmem_get8_nbi(dst_void, src_void, count, target_pe);
  nvshmem_get16_nbi(dst_void, src_void, count, target_pe);
  nvshmem_get32_nbi(dst_void, src_void, count, target_pe);
  nvshmem_get64_nbi(dst_void, src_void, count, target_pe);
  nvshmem_get128_nbi(dst_void, src_void, count, target_pe);

  // nvshmemx_getSIZE_nbi_block
  // ishmemx_getSIZE_nbi_work_group
  // CHECK: ishmemx_get8_nbi_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get16_nbi_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get32_nbi_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get64_nbi_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  // CHECK-NEXT: ishmemx_get128_nbi_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  nvshmemx_get8_nbi_block(dst_void, src_void, count, target_pe);
  nvshmemx_get16_nbi_block(dst_void, src_void, count, target_pe);
  nvshmemx_get32_nbi_block(dst_void, src_void, count, target_pe);
  nvshmemx_get64_nbi_block(dst_void, src_void, count, target_pe);
  nvshmemx_get128_nbi_block(dst_void, src_void, count, target_pe);

  // nvshmemx_getSIZE_nbi_warp
  // ishmemx_getSIZE_nbi_work_group
  // CHECK: ishmemx_get8_nbi_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get16_nbi_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get32_nbi_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get64_nbi_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  // CHECK-NEXT: ishmemx_get128_nbi_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  nvshmemx_get8_nbi_warp(dst_void, src_void, count, target_pe);
  nvshmemx_get16_nbi_warp(dst_void, src_void, count, target_pe);
  nvshmemx_get32_nbi_warp(dst_void, src_void, count, target_pe);
  nvshmemx_get64_nbi_warp(dst_void, src_void, count, target_pe);
  nvshmemx_get128_nbi_warp(dst_void, src_void, count, target_pe);

  // nvshmem_getmem_nbi
  // ishmem_getmem_nbi
  // CHECK: ishmem_getmem_nbi(dst_void, src_void, count, target_pe);
  nvshmem_getmem_nbi(dst_void, src_void, count, target_pe);

  // nvshmemx_getmem_nbi_block
  // ishmemx_getmem_nbi_work_group
  // CHECK: ishmemx_getmem_nbi_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_work_group<3>());
  nvshmemx_getmem_nbi_block(dst_void, src_void, count, target_pe);

  // nvshmemx_getmem_nbi_warp
  // ishmemx_getmem_nbi_work_group
  // CHECK: ishmemx_getmem_nbi_work_group(dst_void, src_void, count, target_pe, sycl::ext::oneapi::this_work_item::get_sub_group());
  nvshmemx_getmem_nbi_warp(dst_void, src_void, count, target_pe);
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

  // nvshmem_TYPENAME_put_nbi
  // ishmem_TYPENAME_put_nbi
  // CHECK: ishmem_put_nbi(dst_float, src_float, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_double, src_double, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_char, src_char, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_schar, src_schar, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_short, src_short, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_int, src_int, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_long, src_long, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_longlong, src_longlong, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_uchar, src_uchar, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_ushort, src_ushort, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_uint, src_uint, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_ulong, src_ulong, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_ulonglong, src_ulonglong, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_int8, src_int8, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_int16, src_int16, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_int32, src_int32, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_int64, src_int64, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_uint8, src_uint8, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_uint16, src_uint16, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_uint32, src_uint32, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_uint64, src_uint64, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_size, src_size, count, target_pe);
  // CHECK-NEXT: ishmem_put_nbi(dst_ptrdiff, src_ptrdiff, count, target_pe);
  nvshmem_float_put_nbi(dst_float, src_float, count, target_pe);
  nvshmem_double_put_nbi(dst_double, src_double, count, target_pe);
  nvshmem_char_put_nbi(dst_char, src_char, count, target_pe);
  nvshmem_schar_put_nbi(dst_schar, src_schar, count, target_pe);
  nvshmem_short_put_nbi(dst_short, src_short, count, target_pe);
  nvshmem_int_put_nbi(dst_int, src_int, count, target_pe);
  nvshmem_long_put_nbi(dst_long, src_long, count, target_pe);
  nvshmem_longlong_put_nbi(dst_longlong, src_longlong, count, target_pe);
  nvshmem_uchar_put_nbi(dst_uchar, src_uchar, count, target_pe);
  nvshmem_ushort_put_nbi(dst_ushort, src_ushort, count, target_pe);
  nvshmem_uint_put_nbi(dst_uint, src_uint, count, target_pe);
  nvshmem_ulong_put_nbi(dst_ulong, src_ulong, count, target_pe);
  nvshmem_ulonglong_put_nbi(dst_ulonglong, src_ulonglong, count, target_pe);
  nvshmem_int8_put_nbi(dst_int8, src_int8, count, target_pe);
  nvshmem_int16_put_nbi(dst_int16, src_int16, count, target_pe);
  nvshmem_int32_put_nbi(dst_int32, src_int32, count, target_pe);
  nvshmem_int64_put_nbi(dst_int64, src_int64, count, target_pe);
  nvshmem_uint8_put_nbi(dst_uint8, src_uint8, count, target_pe);
  nvshmem_uint16_put_nbi(dst_uint16, src_uint16, count, target_pe);
  nvshmem_uint32_put_nbi(dst_uint32, src_uint32, count, target_pe);
  nvshmem_uint64_put_nbi(dst_uint64, src_uint64, count, target_pe);
  nvshmem_size_put_nbi(dst_size, src_size, count, target_pe);
  nvshmem_ptrdiff_put_nbi(dst_ptrdiff, src_ptrdiff, count, target_pe);

  // nvshmemx_TYPENAME_put_nbi_on_stream
  // ishmemx_TYPENAME_put_nbi_on_queue
  // CHECK: ishmemx_put_nbi_on_queue(dst_float, src_float, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_nbi_on_queue(dst_double, src_double, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_nbi_on_queue(dst_char, src_char, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_nbi_on_queue(dst_schar, src_schar, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_nbi_on_queue(dst_short, src_short, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_nbi_on_queue(dst_int, src_int, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_nbi_on_queue(dst_long, src_long, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_nbi_on_queue(dst_longlong, src_longlong, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_nbi_on_queue(dst_uchar, src_uchar, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_nbi_on_queue(dst_ushort, src_ushort, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_nbi_on_queue(dst_uint, src_uint, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_nbi_on_queue(dst_ulong, src_ulong, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_nbi_on_queue(dst_ulonglong, src_ulonglong, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_nbi_on_queue(dst_int8, src_int8, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_nbi_on_queue(dst_int16, src_int16, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_nbi_on_queue(dst_int32, src_int32, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_nbi_on_queue(dst_int64, src_int64, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_nbi_on_queue(dst_uint8, src_uint8, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_nbi_on_queue(dst_uint16, src_uint16, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_nbi_on_queue(dst_uint32, src_uint32, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_nbi_on_queue(dst_uint64, src_uint64, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_nbi_on_queue(dst_size, src_size, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put_nbi_on_queue(dst_ptrdiff, src_ptrdiff, count, target_pe, *stream);
  nvshmemx_float_put_nbi_on_stream(dst_float, src_float, count, target_pe, stream);
  nvshmemx_double_put_nbi_on_stream(dst_double, src_double, count, target_pe, stream);
  nvshmemx_char_put_nbi_on_stream(dst_char, src_char, count, target_pe, stream);
  nvshmemx_schar_put_nbi_on_stream(dst_schar, src_schar, count, target_pe, stream);
  nvshmemx_short_put_nbi_on_stream(dst_short, src_short, count, target_pe, stream);
  nvshmemx_int_put_nbi_on_stream(dst_int, src_int, count, target_pe, stream);
  nvshmemx_long_put_nbi_on_stream(dst_long, src_long, count, target_pe, stream);
  nvshmemx_longlong_put_nbi_on_stream(dst_longlong, src_longlong, count, target_pe, stream);
  nvshmemx_uchar_put_nbi_on_stream(dst_uchar, src_uchar, count, target_pe, stream);
  nvshmemx_ushort_put_nbi_on_stream(dst_ushort, src_ushort, count, target_pe, stream);
  nvshmemx_uint_put_nbi_on_stream(dst_uint, src_uint, count, target_pe, stream);
  nvshmemx_ulong_put_nbi_on_stream(dst_ulong, src_ulong, count, target_pe, stream);
  nvshmemx_ulonglong_put_nbi_on_stream(dst_ulonglong, src_ulonglong, count, target_pe, stream);
  nvshmemx_int8_put_nbi_on_stream(dst_int8, src_int8, count, target_pe, stream);
  nvshmemx_int16_put_nbi_on_stream(dst_int16, src_int16, count, target_pe, stream);
  nvshmemx_int32_put_nbi_on_stream(dst_int32, src_int32, count, target_pe, stream);
  nvshmemx_int64_put_nbi_on_stream(dst_int64, src_int64, count, target_pe, stream);
  nvshmemx_uint8_put_nbi_on_stream(dst_uint8, src_uint8, count, target_pe, stream);
  nvshmemx_uint16_put_nbi_on_stream(dst_uint16, src_uint16, count, target_pe, stream);
  nvshmemx_uint32_put_nbi_on_stream(dst_uint32, src_uint32, count, target_pe, stream);
  nvshmemx_uint64_put_nbi_on_stream(dst_uint64, src_uint64, count, target_pe, stream);
  nvshmemx_size_put_nbi_on_stream(dst_size, src_size, count, target_pe, stream);
  nvshmemx_ptrdiff_put_nbi_on_stream(dst_ptrdiff, src_ptrdiff, count, target_pe, stream);

  // nvshmem_putSIZE_nbi
  // ishmem_putSIZE_nbi
  // CHECK: ishmem_put8_nbi(dst_void, src_void, count, target_pe);
  // CHECK-NEXT: ishmem_put16_nbi(dst_void, src_void, count, target_pe);
  // CHECK-NEXT: ishmem_put32_nbi(dst_void, src_void, count, target_pe);
  // CHECK-NEXT: ishmem_put64_nbi(dst_void, src_void, count, target_pe);
  // CHECK-NEXT: ishmem_put128_nbi(dst_void, src_void, count, target_pe);
  nvshmem_put8_nbi(dst_void, src_void, count, target_pe);
  nvshmem_put16_nbi(dst_void, src_void, count, target_pe);
  nvshmem_put32_nbi(dst_void, src_void, count, target_pe);
  nvshmem_put64_nbi(dst_void, src_void, count, target_pe);
  nvshmem_put128_nbi(dst_void, src_void, count, target_pe);

  // nvshmemx_putSIZE_nbi_on_stream
  // ishmemx_putSIZE_nbi_on_queue
  // CHECK: ishmemx_put8_nbi_on_queue(dst_void, src_void, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put16_nbi_on_queue(dst_void, src_void, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put32_nbi_on_queue(dst_void, src_void, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put64_nbi_on_queue(dst_void, src_void, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_put128_nbi_on_queue(dst_void, src_void, count, target_pe, *stream);
  nvshmemx_put8_nbi_on_stream(dst_void, src_void, count, target_pe, stream);
  nvshmemx_put16_nbi_on_stream(dst_void, src_void, count, target_pe, stream);
  nvshmemx_put32_nbi_on_stream(dst_void, src_void, count, target_pe, stream);
  nvshmemx_put64_nbi_on_stream(dst_void, src_void, count, target_pe, stream);
  nvshmemx_put128_nbi_on_stream(dst_void, src_void, count, target_pe, stream);

  // nvshmem_putmem_nbi
  // ishmem_putmem_nbi
  // CHECK: ishmem_putmem_nbi(dst_void, src_void, count, target_pe);
  nvshmem_putmem_nbi(dst_void, src_void, count, target_pe);

  // nvshmemx_putmem_nbi_on_stream
  // ishmemx_putmem_nbi_on_queue
  // CHECK: ishmemx_putmem_nbi_on_queue(dst_void, src_void, count, target_pe, *stream);
  nvshmemx_putmem_nbi_on_stream(dst_void, src_void, count, target_pe, stream);

  // nvshmem_TYPENAME_get_nbi
  // ishmem_TYPENAME_get_nbi
  // CHECK: ishmem_get_nbi(dst_float, src_float, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_double, src_double, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_char, src_char, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_schar, src_schar, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_short, src_short, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_int, src_int, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_long, src_long, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_longlong, src_longlong, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_uchar, src_uchar, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_ushort, src_ushort, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_uint, src_uint, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_ulong, src_ulong, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_ulonglong, src_ulonglong, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_int8, src_int8, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_int16, src_int16, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_int32, src_int32, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_int64, src_int64, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_uint8, src_uint8, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_uint16, src_uint16, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_uint32, src_uint32, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_uint64, src_uint64, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_size, src_size, count, target_pe);
  // CHECK-NEXT: ishmem_get_nbi(dst_ptrdiff, src_ptrdiff, count, target_pe);
  nvshmem_float_get_nbi(dst_float, src_float, count, target_pe);
  nvshmem_double_get_nbi(dst_double, src_double, count, target_pe);
  nvshmem_char_get_nbi(dst_char, src_char, count, target_pe);
  nvshmem_schar_get_nbi(dst_schar, src_schar, count, target_pe);
  nvshmem_short_get_nbi(dst_short, src_short, count, target_pe);
  nvshmem_int_get_nbi(dst_int, src_int, count, target_pe);
  nvshmem_long_get_nbi(dst_long, src_long, count, target_pe);
  nvshmem_longlong_get_nbi(dst_longlong, src_longlong, count, target_pe);
  nvshmem_uchar_get_nbi(dst_uchar, src_uchar, count, target_pe);
  nvshmem_ushort_get_nbi(dst_ushort, src_ushort, count, target_pe);
  nvshmem_uint_get_nbi(dst_uint, src_uint, count, target_pe);
  nvshmem_ulong_get_nbi(dst_ulong, src_ulong, count, target_pe);
  nvshmem_ulonglong_get_nbi(dst_ulonglong, src_ulonglong, count, target_pe);
  nvshmem_int8_get_nbi(dst_int8, src_int8, count, target_pe);
  nvshmem_int16_get_nbi(dst_int16, src_int16, count, target_pe);
  nvshmem_int32_get_nbi(dst_int32, src_int32, count, target_pe);
  nvshmem_int64_get_nbi(dst_int64, src_int64, count, target_pe);
  nvshmem_uint8_get_nbi(dst_uint8, src_uint8, count, target_pe);
  nvshmem_uint16_get_nbi(dst_uint16, src_uint16, count, target_pe);
  nvshmem_uint32_get_nbi(dst_uint32, src_uint32, count, target_pe);
  nvshmem_uint64_get_nbi(dst_uint64, src_uint64, count, target_pe);
  nvshmem_size_get_nbi(dst_size, src_size, count, target_pe);
  nvshmem_ptrdiff_get_nbi(dst_ptrdiff, src_ptrdiff, count, target_pe);

  // nvshmemx_TYPENAME_get_nbi_on_stream
  // ishmemx_TYPENAME_get_nbi_on_queue
  // CHECK: ishmemx_get_nbi_on_queue(dst_float, src_float, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_nbi_on_queue(dst_double, src_double, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_nbi_on_queue(dst_char, src_char, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_nbi_on_queue(dst_schar, src_schar, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_nbi_on_queue(dst_short, src_short, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_nbi_on_queue(dst_int, src_int, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_nbi_on_queue(dst_long, src_long, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_nbi_on_queue(dst_longlong, src_longlong, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_nbi_on_queue(dst_uchar, src_uchar, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_nbi_on_queue(dst_ushort, src_ushort, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_nbi_on_queue(dst_uint, src_uint, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_nbi_on_queue(dst_ulong, src_ulong, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_nbi_on_queue(dst_ulonglong, src_ulonglong, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_nbi_on_queue(dst_int8, src_int8, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_nbi_on_queue(dst_int16, src_int16, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_nbi_on_queue(dst_int32, src_int32, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_nbi_on_queue(dst_int64, src_int64, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_nbi_on_queue(dst_uint8, src_uint8, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_nbi_on_queue(dst_uint16, src_uint16, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_nbi_on_queue(dst_uint32, src_uint32, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_nbi_on_queue(dst_uint64, src_uint64, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_nbi_on_queue(dst_size, src_size, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get_nbi_on_queue(dst_ptrdiff, src_ptrdiff, count, target_pe, *stream);
  nvshmemx_float_get_nbi_on_stream(dst_float, src_float, count, target_pe, stream);
  nvshmemx_double_get_nbi_on_stream(dst_double, src_double, count, target_pe, stream);
  nvshmemx_char_get_nbi_on_stream(dst_char, src_char, count, target_pe, stream);
  nvshmemx_schar_get_nbi_on_stream(dst_schar, src_schar, count, target_pe, stream);
  nvshmemx_short_get_nbi_on_stream(dst_short, src_short, count, target_pe, stream);
  nvshmemx_int_get_nbi_on_stream(dst_int, src_int, count, target_pe, stream);
  nvshmemx_long_get_nbi_on_stream(dst_long, src_long, count, target_pe, stream);
  nvshmemx_longlong_get_nbi_on_stream(dst_longlong, src_longlong, count, target_pe, stream);
  nvshmemx_uchar_get_nbi_on_stream(dst_uchar, src_uchar, count, target_pe, stream);
  nvshmemx_ushort_get_nbi_on_stream(dst_ushort, src_ushort, count, target_pe, stream);
  nvshmemx_uint_get_nbi_on_stream(dst_uint, src_uint, count, target_pe, stream);
  nvshmemx_ulong_get_nbi_on_stream(dst_ulong, src_ulong, count, target_pe, stream);
  nvshmemx_ulonglong_get_nbi_on_stream(dst_ulonglong, src_ulonglong, count, target_pe, stream);
  nvshmemx_int8_get_nbi_on_stream(dst_int8, src_int8, count, target_pe, stream);
  nvshmemx_int16_get_nbi_on_stream(dst_int16, src_int16, count, target_pe, stream);
  nvshmemx_int32_get_nbi_on_stream(dst_int32, src_int32, count, target_pe, stream);
  nvshmemx_int64_get_nbi_on_stream(dst_int64, src_int64, count, target_pe, stream);
  nvshmemx_uint8_get_nbi_on_stream(dst_uint8, src_uint8, count, target_pe, stream);
  nvshmemx_uint16_get_nbi_on_stream(dst_uint16, src_uint16, count, target_pe, stream);
  nvshmemx_uint32_get_nbi_on_stream(dst_uint32, src_uint32, count, target_pe, stream);
  nvshmemx_uint64_get_nbi_on_stream(dst_uint64, src_uint64, count, target_pe, stream);
  nvshmemx_size_get_nbi_on_stream(dst_size, src_size, count, target_pe, stream);
  nvshmemx_ptrdiff_get_nbi_on_stream(dst_ptrdiff, src_ptrdiff, count, target_pe, stream);

  // nvshmem_getSIZE_nbi
  // ishmem_getSIZE_nbi
  // CHECK: ishmem_get8_nbi(dst_void, src_void, count, target_pe);
  // CHECK-NEXT: ishmem_get16_nbi(dst_void, src_void, count, target_pe);
  // CHECK-NEXT: ishmem_get32_nbi(dst_void, src_void, count, target_pe);
  // CHECK-NEXT: ishmem_get64_nbi(dst_void, src_void, count, target_pe);
  // CHECK-NEXT: ishmem_get128_nbi(dst_void, src_void, count, target_pe);
  nvshmem_get8_nbi(dst_void, src_void, count, target_pe);
  nvshmem_get16_nbi(dst_void, src_void, count, target_pe);
  nvshmem_get32_nbi(dst_void, src_void, count, target_pe);
  nvshmem_get64_nbi(dst_void, src_void, count, target_pe);
  nvshmem_get128_nbi(dst_void, src_void, count, target_pe);

  // nvshmemx_getSIZE_nbi_on_stream
  // ishmemx_getSIZE_nbi_on_queue
  // CHECK: ishmemx_get8_nbi_on_queue(dst_void, src_void, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get16_nbi_on_queue(dst_void, src_void, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get32_nbi_on_queue(dst_void, src_void, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get64_nbi_on_queue(dst_void, src_void, count, target_pe, *stream);
  // CHECK-NEXT: ishmemx_get128_nbi_on_queue(dst_void, src_void, count, target_pe, *stream);
  nvshmemx_get8_nbi_on_stream(dst_void, src_void, count, target_pe, stream);
  nvshmemx_get16_nbi_on_stream(dst_void, src_void, count, target_pe, stream);
  nvshmemx_get32_nbi_on_stream(dst_void, src_void, count, target_pe, stream);
  nvshmemx_get64_nbi_on_stream(dst_void, src_void, count, target_pe, stream);
  nvshmemx_get128_nbi_on_stream(dst_void, src_void, count, target_pe, stream);

  // nvshmem_getmem_nbi
  // ishmem_getmem_nbi
  // CHECK: ishmem_getmem_nbi(dst_void, src_void, count, target_pe);
  nvshmem_getmem_nbi(dst_void, src_void, count, target_pe);

  // nvshmemx_getmem_nbi_on_stream
  // ishmemx_getmem_nbi_on_queue
  // CHECK: ishmemx_getmem_nbi_on_queue(dst_void, src_void, count, target_pe, *stream);
  nvshmemx_getmem_nbi_on_stream(dst_void, src_void, count, target_pe, stream);

  return 0;
}
