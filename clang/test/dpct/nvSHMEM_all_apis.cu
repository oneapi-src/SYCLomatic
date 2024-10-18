// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8, cuda-12.0, cuda-12.1, cuda-12.2, cuda-12.3, cuda-12.4, cuda-12.5, cuda-12.6
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8, v12.0, v12.1, v12.2, v12.3, v12.4, v12.5, v12.6
// RUN: dpct --format-range=none -out-root %T/nvSHMEM %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/nvSHMEM/nvSHMEM_all_apis.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -DNO_BUILD_TEST -fsycl %T/nvSHMEM_all_apis/nvSHMEM_all_apis.dp.cpp -o %T/nvSHMEM_all_apis/nvSHMEM_all_apis.dp.o %}

#include <iostream>
#include <nvshmem.h>

void h_test() {
  float float_var;
  double double_var;
  char char_var;
  signed char schar_var;
  short short_var;
  int int_var;
  long long_var;
  long long longlong_var;
  unsigned char uchar_var;
  unsigned short ushort_var;
  unsigned int uint_var;
  unsigned long ulong_var;
  unsigned long long ulonglong_var;
  int8_t int8_var;
  int16_t int16_var;
  int32_t int32_var;
  int64_t int64_var;
  uint8_t uint8_var;
  uint16_t uint16_var;
  uint32_t uint32_var;
  uint64_t uint64_var;
  size_t size_var;
  ptrdiff_t ptrdiff_var;
  nvshmem_team_t nvshmem_team_t_var;
  nvshmemx_uniqueid_t nvshmemx_uniqueid_t_var;
  const unsigned int const_unsigned_int_var{0};
  void *void_ptr_var;
  const int64_t const_int64_t_var{0};
  const uint8_t const_uint8_t_var{0};
  const unsigned long long const_unsigned_long_long_var{0};
  const ptrdiff_t const_ptrdiff_t_var{0};
  CUmodule CUmodule_var;
  const short const_short_var{0};
  const char const_char_var{0};
  nvshmem_team_config_t nvshmem_team_config_t_var;
  nvshmemx_init_attr_t nvshmemx_init_attr_t_var;
  const void *const_void_var{0};
  const int32_t const_int32_t_var{0};
  const unsigned short const_unsigned_short_var{0};
  const int16_t const_int16_t_var{0};
  const uint32_t const_uint32_t_var{0};
  const double const_double_var{0};
  const unsigned char const_unsigned_char_var{0};
  const long const_long_var{0};
  const int8_t const_int8_t_var{0};
  const long long const_long_long_var{0};
  const nvshmem_team_config_t const_nvshmem_team_config_t_var{0};
  cudaStream_t cudaStream_t_var;
  const signed char const_signed_char_var{0};
  const uint64_t const_uint64_t_var{0};
  const uint16_t const_uint16_t_var{0};
  const float const_float_var{0};
  const unsigned long const_unsigned_long_var{0};
  const size_t const_size_t_var{0};
  const int const_int_var{0};
  dim3 dim3_var;

  /// Library Setup, Exit and Query Routines
  nvshmem_init();
  nvshmemx_init_attr(uint_var, &nvshmemx_init_attr_t_var);
  nvshmemx_hostlib_init_attr(uint_var, &nvshmemx_init_attr_t_var);
  nvshmemx_get_uniqueid(&nvshmemx_uniqueid_t_var);
  nvshmemx_set_attr_uniqueid_args(int_var, int_var, &nvshmemx_uniqueid_t_var, &nvshmemx_init_attr_t_var);
  nvshmemx_cumodule_init(CUmodule_var);
  nvshmemx_init_status();
  nvshmem_my_pe();
  nvshmem_n_pes();
  nvshmem_finalize();
  nvshmem_global_exit(int_var);
  nvshmem_ptr(const_void_var, int_var);
  nvshmem_info_get_version(&int_var, &int_var);
  nvshmem_info_get_name(&char_var);
  nvshmemx_vendor_get_version_info(&int_var, &int_var, &int_var);

  nvshmem_init_thread(int_var, &int_var);
  nvshmem_query_thread(&int_var);

  nvshmemx_collective_launch(&void_ptr_var, dim3_var, dim3_var, &void_ptr_var, size_var, cudaStream_t_var);
  nvshmemx_collective_launch_query_gridsize(&void_ptr_var, dim3_var, &void_ptr_var, size_var, &int_var);


  /// Memory Management
  nvshmem_align(size_var, size_var);
  nvshmem_calloc(size_var, size_var);
  nvshmem_malloc(size_var);
  nvshmem_free(void_ptr_var);

  nvshmemx_buffer_register(void_ptr_var, size_var);
  nvshmemx_buffer_unregister(void_ptr_var);
  nvshmemx_buffer_unregister_all();


  /// Team Management Routines
  nvshmem_team_get_config(nvshmem_team_t_var, &nvshmem_team_config_t_var);

  nvshmem_team_split_strided(nvshmem_team_t_var, int_var, int_var, int_var, &const_nvshmem_team_config_t_var, long_var, &nvshmem_team_t_var);
  nvshmem_team_split_2d(nvshmem_team_t_var, int_var, &const_nvshmem_team_config_t_var, long_var, &nvshmem_team_t_var, &const_nvshmem_team_config_t_var, long_var, &nvshmem_team_t_var);
  nvshmem_team_destroy(nvshmem_team_t_var);

  // nvshmemx_TYPENAME_put_on_stream (RMA)
  nvshmemx_float_put_on_stream(&float_var, &const_float_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_double_put_on_stream(&double_var, &const_double_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_char_put_on_stream(&char_var, &const_char_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_schar_put_on_stream(&schar_var, &const_signed_char_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_short_put_on_stream(&short_var, &const_short_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_int_put_on_stream(&int_var, &const_int_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_long_put_on_stream(&long_var, &const_long_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_longlong_put_on_stream(&longlong_var, &const_long_long_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uchar_put_on_stream(&uchar_var, &const_unsigned_char_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_ushort_put_on_stream(&ushort_var, &const_unsigned_short_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uint_put_on_stream(&uint_var, &const_unsigned_int_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_ulong_put_on_stream(&ulong_var, &const_unsigned_long_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_ulonglong_put_on_stream(&ulonglong_var, &const_unsigned_long_long_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_int8_put_on_stream(&int8_var, &const_int8_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_int16_put_on_stream(&int16_var, &const_int16_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_int32_put_on_stream(&int32_var, &const_int32_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_int64_put_on_stream(&int64_var, &const_int64_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uint8_put_on_stream(&uint8_var, &const_uint8_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uint16_put_on_stream(&uint16_var, &const_uint16_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uint32_put_on_stream(&uint32_var, &const_uint32_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uint64_put_on_stream(&uint64_var, &const_uint64_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_size_put_on_stream(&size_var, &const_size_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_ptrdiff_put_on_stream(&ptrdiff_var, &const_ptrdiff_t_var, size_var, int_var, cudaStream_t_var);

  // nvshmemx_putSIZE_on_stream (8, 16, 32, 64, 128)
  nvshmemx_put8_on_stream(void_ptr_var, const_void_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_put16_on_stream(void_ptr_var, const_void_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_put32_on_stream(void_ptr_var, const_void_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_put64_on_stream(void_ptr_var, const_void_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_put128_on_stream(void_ptr_var, const_void_var, size_var, int_var, cudaStream_t_var);

  nvshmemx_putmem_on_stream(void_ptr_var, const_void_var, size_var, int_var, cudaStream_t_var);

  // nvshmemx_TYPENAME_iput_on_stream (RMA)
  nvshmemx_float_iput_on_stream(&float_var, &const_float_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_double_iput_on_stream(&double_var, &const_double_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_char_iput_on_stream(&char_var, &const_char_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_schar_iput_on_stream(&schar_var, &const_signed_char_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_short_iput_on_stream(&short_var, &const_short_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_int_iput_on_stream(&int_var, &const_int_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_long_iput_on_stream(&long_var, &const_long_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_longlong_iput_on_stream(&longlong_var, &const_long_long_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uchar_iput_on_stream(&uchar_var, &const_unsigned_char_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_ushort_iput_on_stream(&ushort_var, &const_unsigned_short_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uint_iput_on_stream(&uint_var, &const_unsigned_int_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_ulong_iput_on_stream(&ulong_var, &const_unsigned_long_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_ulonglong_iput_on_stream(&ulonglong_var, &const_unsigned_long_long_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_int8_iput_on_stream(&int8_var, &const_int8_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_int16_iput_on_stream(&int16_var, &const_int16_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_int32_iput_on_stream(&int32_var, &const_int32_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_int64_iput_on_stream(&int64_var, &const_int64_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uint8_iput_on_stream(&uint8_var, &const_uint8_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uint16_iput_on_stream(&uint16_var, &const_uint16_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uint32_iput_on_stream(&uint32_var, &const_uint32_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uint64_iput_on_stream(&uint64_var, &const_uint64_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_size_iput_on_stream(&size_var, &const_size_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_ptrdiff_iput_on_stream(&ptrdiff_var, &const_ptrdiff_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);

  // nvshmemx_iputSIZE_on_stream (8, 16, 32, 64, 128)
  nvshmemx_iput8_on_stream(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_iput16_on_stream(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_iput32_on_stream(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_iput64_on_stream(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_iput128_on_stream(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);

  // nvshmemx_TYPENAME_get_on_stream (RMA)
  nvshmemx_float_get_on_stream(&float_var, &const_float_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_double_get_on_stream(&double_var, &const_double_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_char_get_on_stream(&char_var, &const_char_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_schar_get_on_stream(&schar_var, &const_signed_char_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_short_get_on_stream(&short_var, &const_short_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_int_get_on_stream(&int_var, &const_int_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_long_get_on_stream(&long_var, &const_long_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_longlong_get_on_stream(&longlong_var, &const_long_long_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uchar_get_on_stream(&uchar_var, &const_unsigned_char_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_ushort_get_on_stream(&ushort_var, &const_unsigned_short_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uint_get_on_stream(&uint_var, &const_unsigned_int_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_ulong_get_on_stream(&ulong_var, &const_unsigned_long_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_ulonglong_get_on_stream(&ulonglong_var, &const_unsigned_long_long_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_int8_get_on_stream(&int8_var, &const_int8_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_int16_get_on_stream(&int16_var, &const_int16_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_int32_get_on_stream(&int32_var, &const_int32_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_int64_get_on_stream(&int64_var, &const_int64_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uint8_get_on_stream(&uint8_var, &const_uint8_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uint16_get_on_stream(&uint16_var, &const_uint16_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uint32_get_on_stream(&uint32_var, &const_uint32_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uint64_get_on_stream(&uint64_var, &const_uint64_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_size_get_on_stream(&size_var, &const_size_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_ptrdiff_get_on_stream(&ptrdiff_var, &const_ptrdiff_t_var, size_var, int_var, cudaStream_t_var);

  // nvshmemx_getSIZE_on_stream (8, 16, 32, 64, 128)
  nvshmemx_get8_on_stream(void_ptr_var, const_void_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_get16_on_stream(void_ptr_var, const_void_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_get32_on_stream(void_ptr_var, const_void_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_get64_on_stream(void_ptr_var, const_void_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_get128_on_stream(void_ptr_var, const_void_var, size_var, int_var, cudaStream_t_var);

  nvshmemx_getmem_on_stream(void_ptr_var, const_void_var, size_var, int_var, cudaStream_t_var);

  // nvshmemx_TYPENAME_iget_on_stream (RMA)
  nvshmemx_float_iget_on_stream(&float_var, &const_float_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_double_iget_on_stream(&double_var, &const_double_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_char_iget_on_stream(&char_var, &const_char_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_schar_iget_on_stream(&schar_var, &const_signed_char_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_short_iget_on_stream(&short_var, &const_short_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_int_iget_on_stream(&int_var, &const_int_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_long_iget_on_stream(&long_var, &const_long_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_longlong_iget_on_stream(&longlong_var, &const_long_long_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uchar_iget_on_stream(&uchar_var, &const_unsigned_char_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_ushort_iget_on_stream(&ushort_var, &const_unsigned_short_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uint_iget_on_stream(&uint_var, &const_unsigned_int_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_ulong_iget_on_stream(&ulong_var, &const_unsigned_long_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_ulonglong_iget_on_stream(&ulonglong_var, &const_unsigned_long_long_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_int8_iget_on_stream(&int8_var, &const_int8_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_int16_iget_on_stream(&int16_var, &const_int16_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_int32_iget_on_stream(&int32_var, &const_int32_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_int64_iget_on_stream(&int64_var, &const_int64_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uint8_iget_on_stream(&uint8_var, &const_uint8_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uint16_iget_on_stream(&uint16_var, &const_uint16_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uint32_iget_on_stream(&uint32_var, &const_uint32_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uint64_iget_on_stream(&uint64_var, &const_uint64_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_size_iget_on_stream(&size_var, &const_size_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_ptrdiff_iget_on_stream(&ptrdiff_var, &const_ptrdiff_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);

  // nvshmemx_igetSIZE_on_stream (8, 16, 32, 64, 128)
  nvshmemx_iget8_on_stream(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_iget16_on_stream(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_iget32_on_stream(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_iget64_on_stream(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_iget128_on_stream(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var, cudaStream_t_var);

  // nvshmemx_TYPENAME_put_nbi_on_stream (RMA)
  nvshmemx_float_put_nbi_on_stream(&float_var, &const_float_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_double_put_nbi_on_stream(&double_var, &const_double_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_char_put_nbi_on_stream(&char_var, &const_char_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_schar_put_nbi_on_stream(&schar_var, &const_signed_char_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_short_put_nbi_on_stream(&short_var, &const_short_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_int_put_nbi_on_stream(&int_var, &const_int_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_long_put_nbi_on_stream(&long_var, &const_long_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_longlong_put_nbi_on_stream(&longlong_var, &const_long_long_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uchar_put_nbi_on_stream(&uchar_var, &const_unsigned_char_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_ushort_put_nbi_on_stream(&ushort_var, &const_unsigned_short_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uint_put_nbi_on_stream(&uint_var, &const_unsigned_int_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_ulong_put_nbi_on_stream(&ulong_var, &const_unsigned_long_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_ulonglong_put_nbi_on_stream(&ulonglong_var, &const_unsigned_long_long_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_int8_put_nbi_on_stream(&int8_var, &const_int8_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_int16_put_nbi_on_stream(&int16_var, &const_int16_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_int32_put_nbi_on_stream(&int32_var, &const_int32_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_int64_put_nbi_on_stream(&int64_var, &const_int64_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uint8_put_nbi_on_stream(&uint8_var, &const_uint8_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uint16_put_nbi_on_stream(&uint16_var, &const_uint16_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uint32_put_nbi_on_stream(&uint32_var, &const_uint32_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uint64_put_nbi_on_stream(&uint64_var, &const_uint64_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_size_put_nbi_on_stream(&size_var, &const_size_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_ptrdiff_put_nbi_on_stream(&ptrdiff_var, &const_ptrdiff_t_var, size_var, int_var, cudaStream_t_var);

  // nvshmemx_putSIZE_nbi_on_stream (8, 16, 32, 64, 128)
  nvshmemx_put8_nbi_on_stream(void_ptr_var, const_void_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_put16_nbi_on_stream(void_ptr_var, const_void_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_put32_nbi_on_stream(void_ptr_var, const_void_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_put64_nbi_on_stream(void_ptr_var, const_void_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_put128_nbi_on_stream(void_ptr_var, const_void_var, size_var, int_var, cudaStream_t_var);

  nvshmemx_putmem_nbi_on_stream(void_ptr_var, const_void_var, size_var, int_var, cudaStream_t_var);

  // nvshmemx_TYPENAME_get_nbi_on_stream (RMA)
  nvshmemx_float_get_nbi_on_stream(&float_var, &const_float_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_double_get_nbi_on_stream(&double_var, &const_double_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_char_get_nbi_on_stream(&char_var, &const_char_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_schar_get_nbi_on_stream(&schar_var, &const_signed_char_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_short_get_nbi_on_stream(&short_var, &const_short_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_int_get_nbi_on_stream(&int_var, &const_int_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_long_get_nbi_on_stream(&long_var, &const_long_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_longlong_get_nbi_on_stream(&longlong_var, &const_long_long_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uchar_get_nbi_on_stream(&uchar_var, &const_unsigned_char_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_ushort_get_nbi_on_stream(&ushort_var, &const_unsigned_short_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uint_get_nbi_on_stream(&uint_var, &const_unsigned_int_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_ulong_get_nbi_on_stream(&ulong_var, &const_unsigned_long_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_ulonglong_get_nbi_on_stream(&ulonglong_var, &const_unsigned_long_long_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_int8_get_nbi_on_stream(&int8_var, &const_int8_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_int16_get_nbi_on_stream(&int16_var, &const_int16_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_int32_get_nbi_on_stream(&int32_var, &const_int32_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_int64_get_nbi_on_stream(&int64_var, &const_int64_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uint8_get_nbi_on_stream(&uint8_var, &const_uint8_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uint16_get_nbi_on_stream(&uint16_var, &const_uint16_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uint32_get_nbi_on_stream(&uint32_var, &const_uint32_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uint64_get_nbi_on_stream(&uint64_var, &const_uint64_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_size_get_nbi_on_stream(&size_var, &const_size_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_ptrdiff_get_nbi_on_stream(&ptrdiff_var, &const_ptrdiff_t_var, size_var, int_var, cudaStream_t_var);

  // nvshmemx_getSIZE_nbi_on_stream (8, 16, 32, 64, 128)
  nvshmemx_get8_nbi_on_stream(void_ptr_var, const_void_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_get16_nbi_on_stream(void_ptr_var, const_void_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_get32_nbi_on_stream(void_ptr_var, const_void_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_get64_nbi_on_stream(void_ptr_var, const_void_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_get128_nbi_on_stream(void_ptr_var, const_void_var, size_var, int_var, cudaStream_t_var);

  nvshmemx_getmem_nbi_on_stream(void_ptr_var, const_void_var, size_var, int_var, cudaStream_t_var);

  // nvshmemx_TYPENAME_alltoall_on_stream (RMA)
  nvshmemx_float_alltoall_on_stream(nvshmem_team_t_var, &float_var, &const_float_var, size_var, cudaStream_t_var);
  nvshmemx_double_alltoall_on_stream(nvshmem_team_t_var, &double_var, &const_double_var, size_var, cudaStream_t_var);
  nvshmemx_char_alltoall_on_stream(nvshmem_team_t_var, &char_var, &const_char_var, size_var, cudaStream_t_var);
  nvshmemx_schar_alltoall_on_stream(nvshmem_team_t_var, &schar_var, &const_signed_char_var, size_var, cudaStream_t_var);
  nvshmemx_short_alltoall_on_stream(nvshmem_team_t_var, &short_var, &const_short_var, size_var, cudaStream_t_var);
  nvshmemx_int_alltoall_on_stream(nvshmem_team_t_var, &int_var, &const_int_var, size_var, cudaStream_t_var);
  nvshmemx_long_alltoall_on_stream(nvshmem_team_t_var, &long_var, &const_long_var, size_var, cudaStream_t_var);
  nvshmemx_longlong_alltoall_on_stream(nvshmem_team_t_var, &longlong_var, &const_long_long_var, size_var, cudaStream_t_var);
  nvshmemx_uchar_alltoall_on_stream(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var, cudaStream_t_var);
  nvshmemx_ushort_alltoall_on_stream(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var, cudaStream_t_var);
  nvshmemx_uint_alltoall_on_stream(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var, cudaStream_t_var);
  nvshmemx_ulong_alltoall_on_stream(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var, cudaStream_t_var);
  nvshmemx_ulonglong_alltoall_on_stream(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var, cudaStream_t_var);
  nvshmemx_int8_alltoall_on_stream(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var, cudaStream_t_var);
  nvshmemx_int16_alltoall_on_stream(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var, cudaStream_t_var);
  nvshmemx_int32_alltoall_on_stream(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var, cudaStream_t_var);
  nvshmemx_int64_alltoall_on_stream(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var, cudaStream_t_var);
  nvshmemx_uint8_alltoall_on_stream(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var, cudaStream_t_var);
  nvshmemx_uint16_alltoall_on_stream(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var, cudaStream_t_var);
  nvshmemx_uint32_alltoall_on_stream(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var, cudaStream_t_var);
  nvshmemx_uint64_alltoall_on_stream(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var, cudaStream_t_var);
  nvshmemx_size_alltoall_on_stream(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var, cudaStream_t_var);
  nvshmemx_ptrdiff_alltoall_on_stream(nvshmem_team_t_var, &ptrdiff_var, &const_ptrdiff_t_var, size_var, cudaStream_t_var);

  // nvshmemx_TYPENAME_broadcast_on_stream (RMA)
  nvshmemx_float_broadcast_on_stream(nvshmem_team_t_var, &float_var, &const_float_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_double_broadcast_on_stream(nvshmem_team_t_var, &double_var, &const_double_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_char_broadcast_on_stream(nvshmem_team_t_var, &char_var, &const_char_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_schar_broadcast_on_stream(nvshmem_team_t_var, &schar_var, &const_signed_char_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_short_broadcast_on_stream(nvshmem_team_t_var, &short_var, &const_short_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_int_broadcast_on_stream(nvshmem_team_t_var, &int_var, &const_int_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_long_broadcast_on_stream(nvshmem_team_t_var, &long_var, &const_long_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_longlong_broadcast_on_stream(nvshmem_team_t_var, &longlong_var, &const_long_long_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uchar_broadcast_on_stream(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_ushort_broadcast_on_stream(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uint_broadcast_on_stream(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_ulong_broadcast_on_stream(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_ulonglong_broadcast_on_stream(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_int8_broadcast_on_stream(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_int16_broadcast_on_stream(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_int32_broadcast_on_stream(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_int64_broadcast_on_stream(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uint8_broadcast_on_stream(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uint16_broadcast_on_stream(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uint32_broadcast_on_stream(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_uint64_broadcast_on_stream(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_size_broadcast_on_stream(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var, int_var, cudaStream_t_var);
  nvshmemx_ptrdiff_broadcast_on_stream(nvshmem_team_t_var, &ptrdiff_var, &const_ptrdiff_t_var, size_var, int_var, cudaStream_t_var);

  // nvshmemx_TYPENAME_fcollect_on_stream (RMA)
  nvshmemx_float_fcollect_on_stream(nvshmem_team_t_var, &float_var, &const_float_var, size_var, cudaStream_t_var);
  nvshmemx_double_fcollect_on_stream(nvshmem_team_t_var, &double_var, &const_double_var, size_var, cudaStream_t_var);
  nvshmemx_char_fcollect_on_stream(nvshmem_team_t_var, &char_var, &const_char_var, size_var, cudaStream_t_var);
  nvshmemx_schar_fcollect_on_stream(nvshmem_team_t_var, &schar_var, &const_signed_char_var, size_var, cudaStream_t_var);
  nvshmemx_short_fcollect_on_stream(nvshmem_team_t_var, &short_var, &const_short_var, size_var, cudaStream_t_var);
  nvshmemx_int_fcollect_on_stream(nvshmem_team_t_var, &int_var, &const_int_var, size_var, cudaStream_t_var);
  nvshmemx_long_fcollect_on_stream(nvshmem_team_t_var, &long_var, &const_long_var, size_var, cudaStream_t_var);
  nvshmemx_longlong_fcollect_on_stream(nvshmem_team_t_var, &longlong_var, &const_long_long_var, size_var, cudaStream_t_var);
  nvshmemx_uchar_fcollect_on_stream(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var, cudaStream_t_var);
  nvshmemx_ushort_fcollect_on_stream(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var, cudaStream_t_var);
  nvshmemx_uint_fcollect_on_stream(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var, cudaStream_t_var);
  nvshmemx_ulong_fcollect_on_stream(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var, cudaStream_t_var);
  nvshmemx_ulonglong_fcollect_on_stream(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var, cudaStream_t_var);
  nvshmemx_int8_fcollect_on_stream(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var, cudaStream_t_var);
  nvshmemx_int16_fcollect_on_stream(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var, cudaStream_t_var);
  nvshmemx_int32_fcollect_on_stream(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var, cudaStream_t_var);
  nvshmemx_int64_fcollect_on_stream(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var, cudaStream_t_var);
  nvshmemx_uint8_fcollect_on_stream(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var, cudaStream_t_var);
  nvshmemx_uint16_fcollect_on_stream(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var, cudaStream_t_var);
  nvshmemx_uint32_fcollect_on_stream(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var, cudaStream_t_var);
  nvshmemx_uint64_fcollect_on_stream(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var, cudaStream_t_var);
  nvshmemx_size_fcollect_on_stream(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var, cudaStream_t_var);
  nvshmemx_ptrdiff_fcollect_on_stream(nvshmem_team_t_var, &ptrdiff_var, &const_ptrdiff_t_var, size_var, cudaStream_t_var);

  // nvshmemx_TYPENAME_and_reduce_on_stream (AOX)
  nvshmemx_uchar_and_reduce_on_stream(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var, cudaStream_t_var);
  nvshmemx_ushort_and_reduce_on_stream(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var, cudaStream_t_var);
  nvshmemx_uint_and_reduce_on_stream(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var, cudaStream_t_var);
  nvshmemx_ulong_and_reduce_on_stream(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var, cudaStream_t_var);
  nvshmemx_ulonglong_and_reduce_on_stream(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var, cudaStream_t_var);
  nvshmemx_int8_and_reduce_on_stream(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var, cudaStream_t_var);
  nvshmemx_int16_and_reduce_on_stream(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var, cudaStream_t_var);
  nvshmemx_int32_and_reduce_on_stream(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var, cudaStream_t_var);
  nvshmemx_int64_and_reduce_on_stream(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var, cudaStream_t_var);
  nvshmemx_uint8_and_reduce_on_stream(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var, cudaStream_t_var);
  nvshmemx_uint16_and_reduce_on_stream(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var, cudaStream_t_var);
  nvshmemx_uint32_and_reduce_on_stream(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var, cudaStream_t_var);
  nvshmemx_uint64_and_reduce_on_stream(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var, cudaStream_t_var);
  nvshmemx_size_and_reduce_on_stream(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var, cudaStream_t_var);

  // nvshmemx_TYPENAME_or_reduce_on_stream (AOX)
  nvshmemx_uchar_or_reduce_on_stream(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var, cudaStream_t_var);
  nvshmemx_ushort_or_reduce_on_stream(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var, cudaStream_t_var);
  nvshmemx_uint_or_reduce_on_stream(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var, cudaStream_t_var);
  nvshmemx_ulong_or_reduce_on_stream(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var, cudaStream_t_var);
  nvshmemx_ulonglong_or_reduce_on_stream(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var, cudaStream_t_var);
  nvshmemx_int8_or_reduce_on_stream(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var, cudaStream_t_var);
  nvshmemx_int16_or_reduce_on_stream(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var, cudaStream_t_var);
  nvshmemx_int32_or_reduce_on_stream(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var, cudaStream_t_var);
  nvshmemx_int64_or_reduce_on_stream(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var, cudaStream_t_var);
  nvshmemx_uint8_or_reduce_on_stream(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var, cudaStream_t_var);
  nvshmemx_uint16_or_reduce_on_stream(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var, cudaStream_t_var);
  nvshmemx_uint32_or_reduce_on_stream(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var, cudaStream_t_var);
  nvshmemx_uint64_or_reduce_on_stream(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var, cudaStream_t_var);
  nvshmemx_size_or_reduce_on_stream(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var, cudaStream_t_var);

  // nvshmemx_TYPENAME_xor_reduce_on_stream (AOX)
  nvshmemx_uchar_xor_reduce_on_stream(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var, cudaStream_t_var);
  nvshmemx_ushort_xor_reduce_on_stream(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var, cudaStream_t_var);
  nvshmemx_uint_xor_reduce_on_stream(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var, cudaStream_t_var);
  nvshmemx_ulong_xor_reduce_on_stream(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var, cudaStream_t_var);
  nvshmemx_ulonglong_xor_reduce_on_stream(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var, cudaStream_t_var);
  nvshmemx_int8_xor_reduce_on_stream(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var, cudaStream_t_var);
  nvshmemx_int16_xor_reduce_on_stream(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var, cudaStream_t_var);
  nvshmemx_int32_xor_reduce_on_stream(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var, cudaStream_t_var);
  nvshmemx_int64_xor_reduce_on_stream(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var, cudaStream_t_var);
  nvshmemx_uint8_xor_reduce_on_stream(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var, cudaStream_t_var);
  nvshmemx_uint16_xor_reduce_on_stream(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var, cudaStream_t_var);
  nvshmemx_uint32_xor_reduce_on_stream(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var, cudaStream_t_var);
  nvshmemx_uint64_xor_reduce_on_stream(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var, cudaStream_t_var);
  nvshmemx_size_xor_reduce_on_stream(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var, cudaStream_t_var);

  // nvshmemx_TYPENAME_max_reduce_on_stream (RMA)
  nvshmemx_float_max_reduce_on_stream(nvshmem_team_t_var, &float_var, &const_float_var, size_var, cudaStream_t_var);
  nvshmemx_double_max_reduce_on_stream(nvshmem_team_t_var, &double_var, &const_double_var, size_var, cudaStream_t_var);
  nvshmemx_char_max_reduce_on_stream(nvshmem_team_t_var, &char_var, &const_char_var, size_var, cudaStream_t_var);
  nvshmemx_schar_max_reduce_on_stream(nvshmem_team_t_var, &schar_var, &const_signed_char_var, size_var, cudaStream_t_var);
  nvshmemx_short_max_reduce_on_stream(nvshmem_team_t_var, &short_var, &const_short_var, size_var, cudaStream_t_var);
  nvshmemx_int_max_reduce_on_stream(nvshmem_team_t_var, &int_var, &const_int_var, size_var, cudaStream_t_var);
  nvshmemx_long_max_reduce_on_stream(nvshmem_team_t_var, &long_var, &const_long_var, size_var, cudaStream_t_var);
  nvshmemx_longlong_max_reduce_on_stream(nvshmem_team_t_var, &longlong_var, &const_long_long_var, size_var, cudaStream_t_var);
  nvshmemx_uchar_max_reduce_on_stream(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var, cudaStream_t_var);
  nvshmemx_ushort_max_reduce_on_stream(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var, cudaStream_t_var);
  nvshmemx_uint_max_reduce_on_stream(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var, cudaStream_t_var);
  nvshmemx_ulong_max_reduce_on_stream(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var, cudaStream_t_var);
  nvshmemx_ulonglong_max_reduce_on_stream(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var, cudaStream_t_var);
  nvshmemx_int8_max_reduce_on_stream(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var, cudaStream_t_var);
  nvshmemx_int16_max_reduce_on_stream(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var, cudaStream_t_var);
  nvshmemx_int32_max_reduce_on_stream(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var, cudaStream_t_var);
  nvshmemx_int64_max_reduce_on_stream(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var, cudaStream_t_var);
  nvshmemx_uint8_max_reduce_on_stream(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var, cudaStream_t_var);
  nvshmemx_uint16_max_reduce_on_stream(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var, cudaStream_t_var);
  nvshmemx_uint32_max_reduce_on_stream(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var, cudaStream_t_var);
  nvshmemx_uint64_max_reduce_on_stream(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var, cudaStream_t_var);
  nvshmemx_size_max_reduce_on_stream(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var, cudaStream_t_var);
  // nvshmemx_ptrdiff_max_reduce_on_stream(nvshmem_team_t_var, &ptrdiff_var, &const_ptrdiff_t_var, size_var, cudaStream_t_var);

  // nvshmemx_TYPENAME_min_reduce_on_stream (RMA)
  nvshmemx_float_min_reduce_on_stream(nvshmem_team_t_var, &float_var, &const_float_var, size_var, cudaStream_t_var);
  nvshmemx_double_min_reduce_on_stream(nvshmem_team_t_var, &double_var, &const_double_var, size_var, cudaStream_t_var);
  nvshmemx_char_min_reduce_on_stream(nvshmem_team_t_var, &char_var, &const_char_var, size_var, cudaStream_t_var);
  nvshmemx_schar_min_reduce_on_stream(nvshmem_team_t_var, &schar_var, &const_signed_char_var, size_var, cudaStream_t_var);
  nvshmemx_short_min_reduce_on_stream(nvshmem_team_t_var, &short_var, &const_short_var, size_var, cudaStream_t_var);
  nvshmemx_int_min_reduce_on_stream(nvshmem_team_t_var, &int_var, &const_int_var, size_var, cudaStream_t_var);
  nvshmemx_long_min_reduce_on_stream(nvshmem_team_t_var, &long_var, &const_long_var, size_var, cudaStream_t_var);
  nvshmemx_longlong_min_reduce_on_stream(nvshmem_team_t_var, &longlong_var, &const_long_long_var, size_var, cudaStream_t_var);
  nvshmemx_uchar_min_reduce_on_stream(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var, cudaStream_t_var);
  nvshmemx_ushort_min_reduce_on_stream(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var, cudaStream_t_var);
  nvshmemx_uint_min_reduce_on_stream(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var, cudaStream_t_var);
  nvshmemx_ulong_min_reduce_on_stream(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var, cudaStream_t_var);
  nvshmemx_ulonglong_min_reduce_on_stream(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var, cudaStream_t_var);
  nvshmemx_int8_min_reduce_on_stream(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var, cudaStream_t_var);
  nvshmemx_int16_min_reduce_on_stream(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var, cudaStream_t_var);
  nvshmemx_int32_min_reduce_on_stream(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var, cudaStream_t_var);
  nvshmemx_int64_min_reduce_on_stream(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var, cudaStream_t_var);
  nvshmemx_uint8_min_reduce_on_stream(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var, cudaStream_t_var);
  nvshmemx_uint16_min_reduce_on_stream(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var, cudaStream_t_var);
  nvshmemx_uint32_min_reduce_on_stream(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var, cudaStream_t_var);
  nvshmemx_uint64_min_reduce_on_stream(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var, cudaStream_t_var);
  nvshmemx_size_min_reduce_on_stream(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var, cudaStream_t_var);
  // nvshmemx_ptrdiff_min_reduce_on_stream(nvshmem_team_t_var, &ptrdiff_var, &const_ptrdiff_t_var, size_var, cudaStream_t_var);

  // nvshmemx_TYPENAME_sum_reduce_on_stream (RMA)
  nvshmemx_float_sum_reduce_on_stream(nvshmem_team_t_var, &float_var, &const_float_var, size_var, cudaStream_t_var);
  nvshmemx_double_sum_reduce_on_stream(nvshmem_team_t_var, &double_var, &const_double_var, size_var, cudaStream_t_var);
  nvshmemx_char_sum_reduce_on_stream(nvshmem_team_t_var, &char_var, &const_char_var, size_var, cudaStream_t_var);
  nvshmemx_schar_sum_reduce_on_stream(nvshmem_team_t_var, &schar_var, &const_signed_char_var, size_var, cudaStream_t_var);
  nvshmemx_short_sum_reduce_on_stream(nvshmem_team_t_var, &short_var, &const_short_var, size_var, cudaStream_t_var);
  nvshmemx_int_sum_reduce_on_stream(nvshmem_team_t_var, &int_var, &const_int_var, size_var, cudaStream_t_var);
  nvshmemx_long_sum_reduce_on_stream(nvshmem_team_t_var, &long_var, &const_long_var, size_var, cudaStream_t_var);
  nvshmemx_longlong_sum_reduce_on_stream(nvshmem_team_t_var, &longlong_var, &const_long_long_var, size_var, cudaStream_t_var);
  nvshmemx_uchar_sum_reduce_on_stream(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var, cudaStream_t_var);
  nvshmemx_ushort_sum_reduce_on_stream(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var, cudaStream_t_var);
  nvshmemx_uint_sum_reduce_on_stream(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var, cudaStream_t_var);
  nvshmemx_ulong_sum_reduce_on_stream(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var, cudaStream_t_var);
  nvshmemx_ulonglong_sum_reduce_on_stream(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var, cudaStream_t_var);
  nvshmemx_int8_sum_reduce_on_stream(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var, cudaStream_t_var);
  nvshmemx_int16_sum_reduce_on_stream(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var, cudaStream_t_var);
  nvshmemx_int32_sum_reduce_on_stream(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var, cudaStream_t_var);
  nvshmemx_int64_sum_reduce_on_stream(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var, cudaStream_t_var);
  nvshmemx_uint8_sum_reduce_on_stream(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var, cudaStream_t_var);
  nvshmemx_uint16_sum_reduce_on_stream(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var, cudaStream_t_var);
  nvshmemx_uint32_sum_reduce_on_stream(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var, cudaStream_t_var);
  nvshmemx_uint64_sum_reduce_on_stream(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var, cudaStream_t_var);
  nvshmemx_size_sum_reduce_on_stream(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var, cudaStream_t_var);
  // nvshmemx_ptrdiff_sum_reduce_on_stream(nvshmem_team_t_var, &ptrdiff_var, &const_ptrdiff_t_var, size_var, cudaStream_t_var);

  // nvshmemx_TYPENAME_prod_reduce_on_stream (RMA)
  nvshmemx_float_prod_reduce_on_stream(nvshmem_team_t_var, &float_var, &const_float_var, size_var, cudaStream_t_var);
  nvshmemx_double_prod_reduce_on_stream(nvshmem_team_t_var, &double_var, &const_double_var, size_var, cudaStream_t_var);
  nvshmemx_char_prod_reduce_on_stream(nvshmem_team_t_var, &char_var, &const_char_var, size_var, cudaStream_t_var);
  nvshmemx_schar_prod_reduce_on_stream(nvshmem_team_t_var, &schar_var, &const_signed_char_var, size_var, cudaStream_t_var);
  nvshmemx_short_prod_reduce_on_stream(nvshmem_team_t_var, &short_var, &const_short_var, size_var, cudaStream_t_var);
  nvshmemx_int_prod_reduce_on_stream(nvshmem_team_t_var, &int_var, &const_int_var, size_var, cudaStream_t_var);
  nvshmemx_long_prod_reduce_on_stream(nvshmem_team_t_var, &long_var, &const_long_var, size_var, cudaStream_t_var);
  nvshmemx_longlong_prod_reduce_on_stream(nvshmem_team_t_var, &longlong_var, &const_long_long_var, size_var, cudaStream_t_var);
  nvshmemx_uchar_prod_reduce_on_stream(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var, cudaStream_t_var);
  nvshmemx_ushort_prod_reduce_on_stream(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var, cudaStream_t_var);
  nvshmemx_uint_prod_reduce_on_stream(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var, cudaStream_t_var);
  nvshmemx_ulong_prod_reduce_on_stream(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var, cudaStream_t_var);
  nvshmemx_ulonglong_prod_reduce_on_stream(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var, cudaStream_t_var);
  nvshmemx_int8_prod_reduce_on_stream(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var, cudaStream_t_var);
  nvshmemx_int16_prod_reduce_on_stream(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var, cudaStream_t_var);
  nvshmemx_int32_prod_reduce_on_stream(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var, cudaStream_t_var);
  nvshmemx_int64_prod_reduce_on_stream(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var, cudaStream_t_var);
  nvshmemx_uint8_prod_reduce_on_stream(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var, cudaStream_t_var);
  nvshmemx_uint16_prod_reduce_on_stream(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var, cudaStream_t_var);
  nvshmemx_uint32_prod_reduce_on_stream(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var, cudaStream_t_var);
  nvshmemx_uint64_prod_reduce_on_stream(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var, cudaStream_t_var);
  nvshmemx_size_prod_reduce_on_stream(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var, cudaStream_t_var);
  // nvshmemx_ptrdiff_prod_reduce_on_stream(nvshmem_team_t_var, &ptrdiff_var, &const_ptrdiff_t_var, size_var, cudaStream_t_var);

  nvshmemx_alltoallmem_on_stream(nvshmem_team_t_var, void_ptr_var, const_void_var, size_var, cudaStream_t_var);

  nvshmemx_broadcastmem_on_stream(nvshmem_team_t_var, void_ptr_var, const_void_var, size_var, int_var, cudaStream_t_var);

  nvshmemx_fcollectmem_on_stream(nvshmem_team_t_var, void_ptr_var, const_void_var, size_var, cudaStream_t_var);

  nvshmemx_team_sync_on_stream(nvshmem_team_t_var, cudaStream_t_var);

  // nvshmemx_TYPENAME_put_signal_on_stream (RMA)
  nvshmemx_float_put_signal_on_stream(&float_var, &const_float_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_double_put_signal_on_stream(&double_var, &const_double_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_char_put_signal_on_stream(&char_var, &const_char_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_schar_put_signal_on_stream(&schar_var, &const_signed_char_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_short_put_signal_on_stream(&short_var, &const_short_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_int_put_signal_on_stream(&int_var, &const_int_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_long_put_signal_on_stream(&long_var, &const_long_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_longlong_put_signal_on_stream(&longlong_var, &const_long_long_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_uchar_put_signal_on_stream(&uchar_var, &const_unsigned_char_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_ushort_put_signal_on_stream(&ushort_var, &const_unsigned_short_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_uint_put_signal_on_stream(&uint_var, &const_unsigned_int_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_ulong_put_signal_on_stream(&ulong_var, &const_unsigned_long_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_ulonglong_put_signal_on_stream(&ulonglong_var, &const_unsigned_long_long_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_int8_put_signal_on_stream(&int8_var, &const_int8_t_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_int16_put_signal_on_stream(&int16_var, &const_int16_t_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_int32_put_signal_on_stream(&int32_var, &const_int32_t_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_int64_put_signal_on_stream(&int64_var, &const_int64_t_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_uint8_put_signal_on_stream(&uint8_var, &const_uint8_t_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_uint16_put_signal_on_stream(&uint16_var, &const_uint16_t_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_uint32_put_signal_on_stream(&uint32_var, &const_uint32_t_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_uint64_put_signal_on_stream(&uint64_var, &const_uint64_t_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_size_put_signal_on_stream(&size_var, &const_size_t_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_ptrdiff_put_signal_on_stream(&ptrdiff_var, &const_ptrdiff_t_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);

  // nvshmemx_putSIZE_signal_on_stream (8, 16, 32, 64, 128)
  nvshmemx_put8_signal_on_stream(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_put16_signal_on_stream(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_put32_signal_on_stream(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_put64_signal_on_stream(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_put128_signal_on_stream(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);

  nvshmemx_putmem_signal_on_stream(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);

  // nvshmemx_TYPENAME_put_signal_nbi_on_stream (RMA)
  nvshmemx_float_put_signal_nbi_on_stream(&float_var, &const_float_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_double_put_signal_nbi_on_stream(&double_var, &const_double_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_char_put_signal_nbi_on_stream(&char_var, &const_char_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_schar_put_signal_nbi_on_stream(&schar_var, &const_signed_char_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_short_put_signal_nbi_on_stream(&short_var, &const_short_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_int_put_signal_nbi_on_stream(&int_var, &const_int_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_long_put_signal_nbi_on_stream(&long_var, &const_long_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_longlong_put_signal_nbi_on_stream(&longlong_var, &const_long_long_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_uchar_put_signal_nbi_on_stream(&uchar_var, &const_unsigned_char_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_ushort_put_signal_nbi_on_stream(&ushort_var, &const_unsigned_short_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_uint_put_signal_nbi_on_stream(&uint_var, &const_unsigned_int_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_ulong_put_signal_nbi_on_stream(&ulong_var, &const_unsigned_long_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_ulonglong_put_signal_nbi_on_stream(&ulonglong_var, &const_unsigned_long_long_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_int8_put_signal_nbi_on_stream(&int8_var, &const_int8_t_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_int16_put_signal_nbi_on_stream(&int16_var, &const_int16_t_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_int32_put_signal_nbi_on_stream(&int32_var, &const_int32_t_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_int64_put_signal_nbi_on_stream(&int64_var, &const_int64_t_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_uint8_put_signal_nbi_on_stream(&uint8_var, &const_uint8_t_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_uint16_put_signal_nbi_on_stream(&uint16_var, &const_uint16_t_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_uint32_put_signal_nbi_on_stream(&uint32_var, &const_uint32_t_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_uint64_put_signal_nbi_on_stream(&uint64_var, &const_uint64_t_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_size_put_signal_nbi_on_stream(&size_var, &const_size_t_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_ptrdiff_put_signal_nbi_on_stream(&ptrdiff_var, &const_ptrdiff_t_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);

  // nvshmemx_putSIZE_signal_nbi_on_stream (8, 16, 32, 64, 128)
  nvshmemx_put8_signal_nbi_on_stream(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_put16_signal_nbi_on_stream(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_put32_signal_nbi_on_stream(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_put64_signal_nbi_on_stream(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_put128_signal_nbi_on_stream(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);

  nvshmemx_putmem_signal_nbi_on_stream(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var, cudaStream_t_var);

  // nvshmemx_TYPENAME_wait_until_on_stream (AMO/SIG)
  nvshmemx_short_wait_until_on_stream(&short_var, int_var, short_var, cudaStream_t_var);
  nvshmemx_int_wait_until_on_stream(&int_var, int_var, int_var, cudaStream_t_var);
  nvshmemx_long_wait_until_on_stream(&long_var, int_var, long_var, cudaStream_t_var);
  nvshmemx_longlong_wait_until_on_stream(&longlong_var, int_var, longlong_var, cudaStream_t_var);
  nvshmemx_ushort_wait_until_on_stream(&ushort_var, int_var, ushort_var, cudaStream_t_var);
  nvshmemx_uint_wait_until_on_stream(&uint_var, int_var, uint_var, cudaStream_t_var);
  nvshmemx_ulong_wait_until_on_stream(&ulong_var, int_var, ulong_var, cudaStream_t_var);
  nvshmemx_ulonglong_wait_until_on_stream(&ulonglong_var, int_var, ulonglong_var, cudaStream_t_var);
  nvshmemx_int32_wait_until_on_stream(&int32_var, int_var, int32_var, cudaStream_t_var);
  nvshmemx_int64_wait_until_on_stream(&int64_var, int_var, int64_var, cudaStream_t_var);
  nvshmemx_uint32_wait_until_on_stream(&uint32_var, int_var, uint32_var, cudaStream_t_var);
  nvshmemx_uint64_wait_until_on_stream(&uint64_var, int_var, uint64_var, cudaStream_t_var);
  nvshmemx_size_wait_until_on_stream(&size_var, int_var, size_var, cudaStream_t_var);
  nvshmemx_ptrdiff_wait_until_on_stream(&ptrdiff_var, int_var, ptrdiff_var, cudaStream_t_var);

  nvshmemx_signal_wait_until_on_stream(&uint64_var, int_var, uint64_var, cudaStream_t_var);

  nvshmemx_barrier_all_on_stream(cudaStream_t_var);

  nvshmemx_barrier_on_stream(nvshmem_team_t_var, cudaStream_t_var);

  nvshmemx_quiet_on_stream(cudaStream_t_var);

  nvshmemx_sync_all_on_stream(cudaStream_t_var);
}

__device__ void d_test() {
  float float_var;
  double double_var;
  char char_var;
  signed char schar_var;
  short short_var;
  int int_var;
  long long_var;
  long long longlong_var;
  unsigned char uchar_var;
  unsigned short ushort_var;
  unsigned int uint_var;
  unsigned long ulong_var;
  unsigned long long ulonglong_var;
  int8_t int8_var;
  int16_t int16_var;
  int32_t int32_var;
  int64_t int64_var;
  uint8_t uint8_var;
  uint16_t uint16_var;
  uint32_t uint32_var;
  uint64_t uint64_var;
  size_t size_var;
  ptrdiff_t ptrdiff_var;
  nvshmem_team_t nvshmem_team_t_var;
  const unsigned int const_unsigned_int_var{0};
  void *void_ptr_var;
  const int64_t const_int64_t_var{0};
  const uint8_t const_uint8_t_var{0};
  const unsigned long long const_unsigned_long_long_var{0};
  const ptrdiff_t const_ptrdiff_t_var{0};
  const short const_short_var{0};
  const char const_char_var{0};
  const void *const_void_var{0};
  const int32_t const_int32_t_var{0};
  const unsigned short const_unsigned_short_var{0};
  const int16_t const_int16_t_var{0};
  const uint32_t const_uint32_t_var{0};
  const double const_double_var{0};
  const unsigned char const_unsigned_char_var{0};
  const long const_long_var{0};
  const int8_t const_int8_t_var{0};
  const long long const_long_long_var{0};
  // cudaStream_t cudaStream_t_var;
  const signed char const_signed_char_var{0};
  const uint64_t const_uint64_t_var{0};
  const uint16_t const_uint16_t_var{0};
  const float const_float_var{0};
  const unsigned long const_unsigned_long_var{0};
  const size_t const_size_t_var{0};
  const int const_int_var{0};

  /// Team Management Routines
  nvshmem_team_my_pe(nvshmem_team_t_var);
  nvshmem_team_n_pes(nvshmem_team_t_var);
  nvshmem_team_translate_pe(nvshmem_team_t_var, int_var, nvshmem_team_t_var);
  nvshmem_team_sync(nvshmem_team_t_var);

  // nvshmem_TYPENAME_atomic_compare_swap (STD AMO)
  nvshmem_int_atomic_compare_swap(&int_var, int_var, int_var, int_var);
  nvshmem_long_atomic_compare_swap(&long_var, long_var, long_var, int_var);
  nvshmem_longlong_atomic_compare_swap(&longlong_var, longlong_var, longlong_var, int_var);
  nvshmem_size_atomic_compare_swap(&size_var, size_var, size_var, int_var);
  nvshmem_ptrdiff_atomic_compare_swap(&ptrdiff_var, ptrdiff_var, ptrdiff_var, int_var);

  // nvshmem_TYPENAME_atomic_fetch (EXT AMO)
  nvshmem_float_atomic_fetch(&const_float_var, int_var);
  nvshmem_double_atomic_fetch(&const_double_var, int_var);
  nvshmem_int_atomic_fetch(&const_int_var, int_var);
  nvshmem_long_atomic_fetch(&const_long_var, int_var);
  nvshmem_longlong_atomic_fetch(&const_long_long_var, int_var);
  nvshmem_size_atomic_fetch(&const_size_t_var, int_var);
  nvshmem_ptrdiff_atomic_fetch(&const_ptrdiff_t_var, int_var);

  // nvshmem_TYPENAME_atomic_fetch_add (STD AMO)
  nvshmem_int_atomic_fetch_add(&int_var, int_var, int_var);
  nvshmem_long_atomic_fetch_add(&long_var, long_var, int_var);
  nvshmem_longlong_atomic_fetch_add(&longlong_var, longlong_var, int_var);
  nvshmem_size_atomic_fetch_add(&size_var, size_var, int_var);
  nvshmem_ptrdiff_atomic_fetch_add(&ptrdiff_var, ptrdiff_var, int_var);
  nvshmem_uint_atomic_fetch_and(&uint_var, uint_var, int_var);
  nvshmem_ulong_atomic_fetch_and(&ulong_var, ulong_var, int_var);
  nvshmem_ulonglong_atomic_fetch_and(&ulonglong_var, ulonglong_var, int_var);
  nvshmem_int32_atomic_fetch_and(&int32_var, int32_var, int_var);
  nvshmem_int64_atomic_fetch_and(&int64_var, int64_var, int_var);
  nvshmem_uint32_atomic_fetch_and(&uint32_var, uint32_var, int_var);
  nvshmem_uint64_atomic_fetch_and(&uint64_var, uint64_var, int_var);

  nvshmem_int_atomic_fetch_inc(&int_var, int_var);
  nvshmem_long_atomic_fetch_inc(&long_var, int_var);
  nvshmem_longlong_atomic_fetch_inc(&longlong_var, int_var);
  nvshmem_size_atomic_fetch_inc(&size_var, int_var);
  nvshmem_ptrdiff_atomic_fetch_inc(&ptrdiff_var, int_var);

  nvshmem_uint_atomic_fetch_or(&uint_var, uint_var, int_var);
  nvshmem_ulong_atomic_fetch_or(&ulong_var, ulong_var, int_var);
  nvshmem_ulonglong_atomic_fetch_or(&ulonglong_var, ulonglong_var, int_var);
  nvshmem_int32_atomic_fetch_or(&int32_var, int32_var, int_var);
  nvshmem_int64_atomic_fetch_or(&int64_var, int64_var, int_var);
  nvshmem_uint32_atomic_fetch_or(&uint32_var, uint32_var, int_var);
  nvshmem_uint64_atomic_fetch_or(&uint64_var, uint64_var, int_var);

  nvshmem_uint_atomic_fetch_xor(&uint_var, uint_var, int_var);
  nvshmem_ulong_atomic_fetch_xor(&ulong_var, ulong_var, int_var);
  nvshmem_ulonglong_atomic_fetch_xor(&ulonglong_var, ulonglong_var, int_var);
  nvshmem_int32_atomic_fetch_xor(&int32_var, int32_var, int_var);
  nvshmem_int64_atomic_fetch_xor(&int64_var, int64_var, int_var);
  nvshmem_uint32_atomic_fetch_xor(&uint32_var, uint32_var, int_var);
  nvshmem_uint64_atomic_fetch_xor(&uint64_var, uint64_var, int_var);

  nvshmem_float_atomic_swap(&float_var, float_var, int_var);
  nvshmem_double_atomic_swap(&double_var, double_var, int_var);
  nvshmem_int_atomic_swap(&int_var, int_var, int_var);
  nvshmem_long_atomic_swap(&long_var, long_var, int_var);
  nvshmem_longlong_atomic_swap(&longlong_var, longlong_var, int_var);
  nvshmem_size_atomic_swap(&size_var, size_var, int_var);
  nvshmem_ptrdiff_atomic_swap(&ptrdiff_var, ptrdiff_var, int_var);

  nvshmem_float_g(&const_float_var, int_var);
  nvshmem_double_g(&const_double_var, int_var);
  nvshmem_char_g(&const_char_var, int_var);
  nvshmem_schar_g(&const_signed_char_var, int_var);
  nvshmem_short_g(&const_short_var, int_var);
  nvshmem_int_g(&const_int_var, int_var);
  nvshmem_long_g(&const_long_var, int_var);
  nvshmem_longlong_g(&const_long_long_var, int_var);
  nvshmem_uchar_g(&const_unsigned_char_var, int_var);
  nvshmem_ushort_g(&const_unsigned_short_var, int_var);
  nvshmem_uint_g(&const_unsigned_int_var, int_var);
  nvshmem_ulong_g(&const_unsigned_long_var, int_var);
  nvshmem_ulonglong_g(&const_unsigned_long_long_var, int_var);
  nvshmem_int8_g(&const_int8_t_var, int_var);
  nvshmem_int16_g(&const_int16_t_var, int_var);
  nvshmem_int32_g(&const_int32_t_var, int_var);
  nvshmem_int64_g(&const_int64_t_var, int_var);
  nvshmem_uint8_g(&const_uint8_t_var, int_var);
  nvshmem_uint16_g(&const_uint16_t_var, int_var);
  nvshmem_uint32_g(&const_uint32_t_var, int_var);
  nvshmem_uint64_g(&const_uint64_t_var, int_var);
  nvshmem_size_g(&const_size_t_var, int_var);
  nvshmem_ptrdiff_g(&const_ptrdiff_t_var, int_var);

  nvshmemx_signal_op(&uint64_var, uint64_var, int_var, int_var);

  nvshmem_float_alltoall(nvshmem_team_t_var, &float_var, &const_float_var, size_var);
  nvshmem_double_alltoall(nvshmem_team_t_var, &double_var, &const_double_var, size_var);
  nvshmem_char_alltoall(nvshmem_team_t_var, &char_var, &const_char_var, size_var);
  nvshmem_schar_alltoall(nvshmem_team_t_var, &schar_var, &const_signed_char_var, size_var);
  nvshmem_short_alltoall(nvshmem_team_t_var, &short_var, &const_short_var, size_var);
  nvshmem_int_alltoall(nvshmem_team_t_var, &int_var, &const_int_var, size_var);
  nvshmem_long_alltoall(nvshmem_team_t_var, &long_var, &const_long_var, size_var);
  nvshmem_longlong_alltoall(nvshmem_team_t_var, &longlong_var, &const_long_long_var, size_var);
  nvshmem_uchar_alltoall(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var);
  nvshmem_ushort_alltoall(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var);
  nvshmem_uint_alltoall(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var);
  nvshmem_ulong_alltoall(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var);
  nvshmem_ulonglong_alltoall(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var);
  nvshmem_int8_alltoall(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var);
  nvshmem_int16_alltoall(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var);
  nvshmem_int32_alltoall(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var);
  nvshmem_int64_alltoall(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var);
  nvshmem_uint8_alltoall(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var);
  nvshmem_uint16_alltoall(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var);
  nvshmem_uint32_alltoall(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var);
  nvshmem_uint64_alltoall(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var);
  nvshmem_size_alltoall(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var);
  nvshmem_ptrdiff_alltoall(nvshmem_team_t_var, &ptrdiff_var, &const_ptrdiff_t_var, size_var);

  nvshmem_uchar_and_reduce(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var);
  nvshmem_ushort_and_reduce(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var);
  nvshmem_uint_and_reduce(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var);
  nvshmem_ulong_and_reduce(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var);
  nvshmem_ulonglong_and_reduce(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var);
  nvshmem_int8_and_reduce(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var);
  nvshmem_int16_and_reduce(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var);
  nvshmem_int32_and_reduce(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var);
  nvshmem_int64_and_reduce(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var);
  nvshmem_uint8_and_reduce(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var);
  nvshmem_uint16_and_reduce(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var);
  nvshmem_uint32_and_reduce(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var);
  nvshmem_uint64_and_reduce(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var);
  nvshmem_size_and_reduce(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var);

  nvshmem_float_broadcast(nvshmem_team_t_var, &float_var, &const_float_var, size_var, int_var);
  nvshmem_double_broadcast(nvshmem_team_t_var, &double_var, &const_double_var, size_var, int_var);
  nvshmem_char_broadcast(nvshmem_team_t_var, &char_var, &const_char_var, size_var, int_var);
  nvshmem_schar_broadcast(nvshmem_team_t_var, &schar_var, &const_signed_char_var, size_var, int_var);
  nvshmem_short_broadcast(nvshmem_team_t_var, &short_var, &const_short_var, size_var, int_var);
  nvshmem_int_broadcast(nvshmem_team_t_var, &int_var, &const_int_var, size_var, int_var);
  nvshmem_long_broadcast(nvshmem_team_t_var, &long_var, &const_long_var, size_var, int_var);
  nvshmem_longlong_broadcast(nvshmem_team_t_var, &longlong_var, &const_long_long_var, size_var, int_var);
  nvshmem_uchar_broadcast(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var, int_var);
  nvshmem_ushort_broadcast(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var, int_var);
  nvshmem_uint_broadcast(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var, int_var);
  nvshmem_ulong_broadcast(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var, int_var);
  nvshmem_ulonglong_broadcast(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var, int_var);
  nvshmem_int8_broadcast(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var, int_var);
  nvshmem_int16_broadcast(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var, int_var);
  nvshmem_int32_broadcast(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var, int_var);
  nvshmem_int64_broadcast(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var, int_var);
  nvshmem_uint8_broadcast(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var, int_var);
  nvshmem_uint16_broadcast(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var, int_var);
  nvshmem_uint32_broadcast(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var, int_var);
  nvshmem_uint64_broadcast(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var, int_var);
  nvshmem_size_broadcast(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var, int_var);
  nvshmem_ptrdiff_broadcast(nvshmem_team_t_var, &ptrdiff_var, &const_ptrdiff_t_var, size_var, int_var);

  nvshmem_float_fcollect(nvshmem_team_t_var, &float_var, &const_float_var, size_var);
  nvshmem_double_fcollect(nvshmem_team_t_var, &double_var, &const_double_var, size_var);
  nvshmem_char_fcollect(nvshmem_team_t_var, &char_var, &const_char_var, size_var);
  nvshmem_schar_fcollect(nvshmem_team_t_var, &schar_var, &const_signed_char_var, size_var);
  nvshmem_short_fcollect(nvshmem_team_t_var, &short_var, &const_short_var, size_var);
  nvshmem_int_fcollect(nvshmem_team_t_var, &int_var, &const_int_var, size_var);
  nvshmem_long_fcollect(nvshmem_team_t_var, &long_var, &const_long_var, size_var);
  nvshmem_longlong_fcollect(nvshmem_team_t_var, &longlong_var, &const_long_long_var, size_var);
  nvshmem_uchar_fcollect(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var);
  nvshmem_ushort_fcollect(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var);
  nvshmem_uint_fcollect(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var);
  nvshmem_ulong_fcollect(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var);
  nvshmem_ulonglong_fcollect(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var);
  nvshmem_int8_fcollect(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var);
  nvshmem_int16_fcollect(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var);
  nvshmem_int32_fcollect(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var);
  nvshmem_int64_fcollect(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var);
  nvshmem_uint8_fcollect(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var);
  nvshmem_uint16_fcollect(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var);
  nvshmem_uint32_fcollect(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var);
  nvshmem_uint64_fcollect(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var);
  nvshmem_size_fcollect(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var);
  nvshmem_ptrdiff_fcollect(nvshmem_team_t_var, &ptrdiff_var, &const_ptrdiff_t_var, size_var);

  nvshmem_float_max_reduce(nvshmem_team_t_var, &float_var, &const_float_var, size_var);
  nvshmem_double_max_reduce(nvshmem_team_t_var, &double_var, &const_double_var, size_var);
  nvshmem_char_max_reduce(nvshmem_team_t_var, &char_var, &const_char_var, size_var);
  nvshmem_schar_max_reduce(nvshmem_team_t_var, &schar_var, &const_signed_char_var, size_var);
  nvshmem_short_max_reduce(nvshmem_team_t_var, &short_var, &const_short_var, size_var);
  nvshmem_int_max_reduce(nvshmem_team_t_var, &int_var, &const_int_var, size_var);
  nvshmem_long_max_reduce(nvshmem_team_t_var, &long_var, &const_long_var, size_var);
  nvshmem_longlong_max_reduce(nvshmem_team_t_var, &longlong_var, &const_long_long_var, size_var);
  nvshmem_uchar_max_reduce(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var);
  nvshmem_ushort_max_reduce(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var);
  nvshmem_uint_max_reduce(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var);
  nvshmem_ulong_max_reduce(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var);
  nvshmem_ulonglong_max_reduce(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var);
  nvshmem_int8_max_reduce(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var);
  nvshmem_int16_max_reduce(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var);
  nvshmem_int32_max_reduce(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var);
  nvshmem_int64_max_reduce(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var);
  nvshmem_uint8_max_reduce(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var);
  nvshmem_uint16_max_reduce(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var);
  nvshmem_uint32_max_reduce(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var);
  nvshmem_uint64_max_reduce(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var);
  nvshmem_size_max_reduce(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var);
  // nvshmem_ptrdiff_max_reduce(nvshmem_team_t_var, &ptrdiff_var, &const_ptrdiff_t_var, size_var);

  nvshmem_float_min_reduce(nvshmem_team_t_var, &float_var, &const_float_var, size_var);
  nvshmem_double_min_reduce(nvshmem_team_t_var, &double_var, &const_double_var, size_var);
  nvshmem_char_min_reduce(nvshmem_team_t_var, &char_var, &const_char_var, size_var);
  nvshmem_schar_min_reduce(nvshmem_team_t_var, &schar_var, &const_signed_char_var, size_var);
  nvshmem_short_min_reduce(nvshmem_team_t_var, &short_var, &const_short_var, size_var);
  nvshmem_int_min_reduce(nvshmem_team_t_var, &int_var, &const_int_var, size_var);
  nvshmem_long_min_reduce(nvshmem_team_t_var, &long_var, &const_long_var, size_var);
  nvshmem_longlong_min_reduce(nvshmem_team_t_var, &longlong_var, &const_long_long_var, size_var);
  nvshmem_uchar_min_reduce(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var);
  nvshmem_ushort_min_reduce(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var);
  nvshmem_uint_min_reduce(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var);
  nvshmem_ulong_min_reduce(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var);
  nvshmem_ulonglong_min_reduce(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var);
  nvshmem_int8_min_reduce(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var);
  nvshmem_int16_min_reduce(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var);
  nvshmem_int32_min_reduce(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var);
  nvshmem_int64_min_reduce(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var);
  nvshmem_uint8_min_reduce(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var);
  nvshmem_uint16_min_reduce(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var);
  nvshmem_uint32_min_reduce(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var);
  nvshmem_uint64_min_reduce(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var);
  nvshmem_size_min_reduce(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var);
  // nvshmem_ptrdiff_min_reduce(nvshmem_team_t_var, &ptrdiff_var, &const_ptrdiff_t_var, size_var);

  nvshmem_uchar_or_reduce(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var);
  nvshmem_ushort_or_reduce(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var);
  nvshmem_uint_or_reduce(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var);
  nvshmem_ulong_or_reduce(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var);
  nvshmem_ulonglong_or_reduce(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var);
  nvshmem_int8_or_reduce(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var);
  nvshmem_int16_or_reduce(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var);
  nvshmem_int32_or_reduce(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var);
  nvshmem_int64_or_reduce(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var);
  nvshmem_uint8_or_reduce(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var);
  nvshmem_uint16_or_reduce(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var);
  nvshmem_uint32_or_reduce(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var);
  nvshmem_uint64_or_reduce(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var);
  nvshmem_size_or_reduce(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var);

  nvshmem_float_prod_reduce(nvshmem_team_t_var, &float_var, &const_float_var, size_var);
  nvshmem_double_prod_reduce(nvshmem_team_t_var, &double_var, &const_double_var, size_var);
  nvshmem_char_prod_reduce(nvshmem_team_t_var, &char_var, &const_char_var, size_var);
  nvshmem_schar_prod_reduce(nvshmem_team_t_var, &schar_var, &const_signed_char_var, size_var);
  nvshmem_short_prod_reduce(nvshmem_team_t_var, &short_var, &const_short_var, size_var);
  nvshmem_int_prod_reduce(nvshmem_team_t_var, &int_var, &const_int_var, size_var);
  nvshmem_long_prod_reduce(nvshmem_team_t_var, &long_var, &const_long_var, size_var);
  nvshmem_longlong_prod_reduce(nvshmem_team_t_var, &longlong_var, &const_long_long_var, size_var);
  nvshmem_uchar_prod_reduce(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var);
  nvshmem_ushort_prod_reduce(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var);
  nvshmem_uint_prod_reduce(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var);
  nvshmem_ulong_prod_reduce(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var);
  nvshmem_ulonglong_prod_reduce(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var);
  nvshmem_int8_prod_reduce(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var);
  nvshmem_int16_prod_reduce(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var);
  nvshmem_int32_prod_reduce(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var);
  nvshmem_int64_prod_reduce(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var);
  nvshmem_uint8_prod_reduce(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var);
  nvshmem_uint16_prod_reduce(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var);
  nvshmem_uint32_prod_reduce(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var);
  nvshmem_uint64_prod_reduce(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var);
  nvshmem_size_prod_reduce(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var);
  // nvshmem_ptrdiff_prod_reduce(nvshmem_team_t_var, &ptrdiff_var, &const_ptrdiff_t_var, size_var);

  nvshmem_float_sum_reduce(nvshmem_team_t_var, &float_var, &const_float_var, size_var);
  nvshmem_double_sum_reduce(nvshmem_team_t_var, &double_var, &const_double_var, size_var);
  nvshmem_char_sum_reduce(nvshmem_team_t_var, &char_var, &const_char_var, size_var);
  nvshmem_schar_sum_reduce(nvshmem_team_t_var, &schar_var, &const_signed_char_var, size_var);
  nvshmem_short_sum_reduce(nvshmem_team_t_var, &short_var, &const_short_var, size_var);
  nvshmem_int_sum_reduce(nvshmem_team_t_var, &int_var, &const_int_var, size_var);
  nvshmem_long_sum_reduce(nvshmem_team_t_var, &long_var, &const_long_var, size_var);
  nvshmem_longlong_sum_reduce(nvshmem_team_t_var, &longlong_var, &const_long_long_var, size_var);
  nvshmem_uchar_sum_reduce(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var);
  nvshmem_ushort_sum_reduce(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var);
  nvshmem_uint_sum_reduce(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var);
  nvshmem_ulong_sum_reduce(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var);
  nvshmem_ulonglong_sum_reduce(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var);
  nvshmem_int8_sum_reduce(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var);
  nvshmem_int16_sum_reduce(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var);
  nvshmem_int32_sum_reduce(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var);
  nvshmem_int64_sum_reduce(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var);
  nvshmem_uint8_sum_reduce(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var);
  nvshmem_uint16_sum_reduce(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var);
  nvshmem_uint32_sum_reduce(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var);
  nvshmem_uint64_sum_reduce(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var);
  nvshmem_size_sum_reduce(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var);
  // nvshmem_ptrdiff_sum_reduce(nvshmem_team_t_var, &ptrdiff_var, &const_ptrdiff_t_var, size_var);

  nvshmem_short_test(&short_var, int_var, short_var);
  nvshmem_int_test(&int_var, int_var, int_var);
  nvshmem_long_test(&long_var, int_var, long_var);
  nvshmem_longlong_test(&longlong_var, int_var, longlong_var);
  nvshmem_ushort_test(&ushort_var, int_var, ushort_var);
  nvshmem_uint_test(&uint_var, int_var, uint_var);
  nvshmem_ulong_test(&ulong_var, int_var, ulong_var);
  nvshmem_ulonglong_test(&ulonglong_var, int_var, ulonglong_var);
  nvshmem_int32_test(&int32_var, int_var, int32_var);
  nvshmem_int64_test(&int64_var, int_var, int64_var);
  nvshmem_uint32_test(&uint32_var, int_var, uint32_var);
  nvshmem_uint64_test(&uint64_var, int_var, uint64_var);
  nvshmem_size_test(&size_var, int_var, size_var);
  nvshmem_ptrdiff_test(&ptrdiff_var, int_var, ptrdiff_var);

  nvshmem_short_test_all(&short_var, size_var, &const_int_var, int_var, short_var);
  nvshmem_int_test_all(&int_var, size_var, &const_int_var, int_var, int_var);
  nvshmem_long_test_all(&long_var, size_var, &const_int_var, int_var, long_var);
  nvshmem_longlong_test_all(&longlong_var, size_var, &const_int_var, int_var, longlong_var);
  nvshmem_ushort_test_all(&ushort_var, size_var, &const_int_var, int_var, ushort_var);
  nvshmem_uint_test_all(&uint_var, size_var, &const_int_var, int_var, uint_var);
  nvshmem_ulong_test_all(&ulong_var, size_var, &const_int_var, int_var, ulong_var);
  nvshmem_ulonglong_test_all(&ulonglong_var, size_var, &const_int_var, int_var, ulonglong_var);
  nvshmem_int32_test_all(&int32_var, size_var, &const_int_var, int_var, int32_var);
  nvshmem_int64_test_all(&int64_var, size_var, &const_int_var, int_var, int64_var);
  nvshmem_uint32_test_all(&uint32_var, size_var, &const_int_var, int_var, uint32_var);
  nvshmem_uint64_test_all(&uint64_var, size_var, &const_int_var, int_var, uint64_var);
  nvshmem_size_test_all(&size_var, size_var, &const_int_var, int_var, size_var);
  nvshmem_ptrdiff_test_all(&ptrdiff_var, size_var, &const_int_var, int_var, ptrdiff_var);

  nvshmem_short_test_all_vector(&short_var, size_var, &const_int_var, int_var, &short_var);
  nvshmem_int_test_all_vector(&int_var, size_var, &const_int_var, int_var, &int_var);
  nvshmem_long_test_all_vector(&long_var, size_var, &const_int_var, int_var, &long_var);
  nvshmem_longlong_test_all_vector(&longlong_var, size_var, &const_int_var, int_var, &longlong_var);
  nvshmem_ushort_test_all_vector(&ushort_var, size_var, &const_int_var, int_var, &ushort_var);
  nvshmem_uint_test_all_vector(&uint_var, size_var, &const_int_var, int_var, &uint_var);
  nvshmem_ulong_test_all_vector(&ulong_var, size_var, &const_int_var, int_var, &ulong_var);
  nvshmem_ulonglong_test_all_vector(&ulonglong_var, size_var, &const_int_var, int_var, &ulonglong_var);
  nvshmem_int32_test_all_vector(&int32_var, size_var, &const_int_var, int_var, &int32_var);
  nvshmem_int64_test_all_vector(&int64_var, size_var, &const_int_var, int_var, &int64_var);
  nvshmem_uint32_test_all_vector(&uint32_var, size_var, &const_int_var, int_var, &uint32_var);
  nvshmem_uint64_test_all_vector(&uint64_var, size_var, &const_int_var, int_var, &uint64_var);
  nvshmem_size_test_all_vector(&size_var, size_var, &const_int_var, int_var, &size_var);
  nvshmem_ptrdiff_test_all_vector(&ptrdiff_var, size_var, &const_int_var, int_var, &ptrdiff_var);

  nvshmem_uchar_xor_reduce(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var);
  nvshmem_ushort_xor_reduce(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var);
  nvshmem_uint_xor_reduce(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var);
  nvshmem_ulong_xor_reduce(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var);
  nvshmem_ulonglong_xor_reduce(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var);
  nvshmem_int8_xor_reduce(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var);
  nvshmem_int16_xor_reduce(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var);
  nvshmem_int32_xor_reduce(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var);
  nvshmem_int64_xor_reduce(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var);
  nvshmem_uint8_xor_reduce(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var);
  nvshmem_uint16_xor_reduce(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var);
  nvshmem_uint32_xor_reduce(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var);
  nvshmem_uint64_xor_reduce(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var);
  nvshmem_size_xor_reduce(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var);

  nvshmem_alltoallmem(nvshmem_team_t_var, void_ptr_var, const_void_var, size_var);
  nvshmem_broadcastmem(nvshmem_team_t_var, void_ptr_var, const_void_var, size_var, int_var);
  nvshmem_fcollectmem(nvshmem_team_t_var, void_ptr_var, const_void_var, size_var);
  nvshmem_sync(nvshmem_team_t_var);

  // nvshmemx_TYPENAME_alltoall_block (RMA)
  nvshmemx_float_alltoall_block(nvshmem_team_t_var, &float_var, &const_float_var, size_var);
  nvshmemx_double_alltoall_block(nvshmem_team_t_var, &double_var, &const_double_var, size_var);
  nvshmemx_char_alltoall_block(nvshmem_team_t_var, &char_var, &const_char_var, size_var);
  nvshmemx_schar_alltoall_block(nvshmem_team_t_var, &schar_var, &const_signed_char_var, size_var);
  nvshmemx_short_alltoall_block(nvshmem_team_t_var, &short_var, &const_short_var, size_var);
  nvshmemx_int_alltoall_block(nvshmem_team_t_var, &int_var, &const_int_var, size_var);
  nvshmemx_long_alltoall_block(nvshmem_team_t_var, &long_var, &const_long_var, size_var);
  nvshmemx_longlong_alltoall_block(nvshmem_team_t_var, &longlong_var, &const_long_long_var, size_var);
  nvshmemx_uchar_alltoall_block(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var);
  nvshmemx_ushort_alltoall_block(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var);
  nvshmemx_uint_alltoall_block(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var);
  nvshmemx_ulong_alltoall_block(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var);
  nvshmemx_ulonglong_alltoall_block(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var);
  nvshmemx_int8_alltoall_block(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var);
  nvshmemx_int16_alltoall_block(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var);
  nvshmemx_int32_alltoall_block(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var);
  nvshmemx_int64_alltoall_block(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var);
  nvshmemx_uint8_alltoall_block(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var);
  nvshmemx_uint16_alltoall_block(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var);
  nvshmemx_uint32_alltoall_block(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var);
  nvshmemx_uint64_alltoall_block(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var);
  nvshmemx_size_alltoall_block(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var);
  nvshmemx_ptrdiff_alltoall_block(nvshmem_team_t_var, &ptrdiff_var, &const_ptrdiff_t_var, size_var);

  // nvshmemx_TYPENAME_alltoall_warp (RMA)
  nvshmemx_float_alltoall_warp(nvshmem_team_t_var, &float_var, &const_float_var, size_var);
  nvshmemx_double_alltoall_warp(nvshmem_team_t_var, &double_var, &const_double_var, size_var);
  nvshmemx_char_alltoall_warp(nvshmem_team_t_var, &char_var, &const_char_var, size_var);
  nvshmemx_schar_alltoall_warp(nvshmem_team_t_var, &schar_var, &const_signed_char_var, size_var);
  nvshmemx_short_alltoall_warp(nvshmem_team_t_var, &short_var, &const_short_var, size_var);
  nvshmemx_int_alltoall_warp(nvshmem_team_t_var, &int_var, &const_int_var, size_var);
  nvshmemx_long_alltoall_warp(nvshmem_team_t_var, &long_var, &const_long_var, size_var);
  nvshmemx_longlong_alltoall_warp(nvshmem_team_t_var, &longlong_var, &const_long_long_var, size_var);
  nvshmemx_uchar_alltoall_warp(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var);
  nvshmemx_ushort_alltoall_warp(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var);
  nvshmemx_uint_alltoall_warp(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var);
  nvshmemx_ulong_alltoall_warp(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var);
  nvshmemx_ulonglong_alltoall_warp(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var);
  nvshmemx_int8_alltoall_warp(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var);
  nvshmemx_int16_alltoall_warp(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var);
  nvshmemx_int32_alltoall_warp(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var);
  nvshmemx_int64_alltoall_warp(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var);
  nvshmemx_uint8_alltoall_warp(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var);
  nvshmemx_uint16_alltoall_warp(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var);
  nvshmemx_uint32_alltoall_warp(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var);
  nvshmemx_uint64_alltoall_warp(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var);
  nvshmemx_size_alltoall_warp(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var);
  nvshmemx_ptrdiff_alltoall_warp(nvshmem_team_t_var, &ptrdiff_var, &const_ptrdiff_t_var, size_var);

  // nvshmemx_TYPENAME_and_reduce_block (AOX)
  nvshmemx_uchar_and_reduce_block(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var);
  nvshmemx_ushort_and_reduce_block(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var);
  nvshmemx_uint_and_reduce_block(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var);
  nvshmemx_ulong_and_reduce_block(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var);
  nvshmemx_ulonglong_and_reduce_block(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var);
  nvshmemx_int8_and_reduce_block(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var);
  nvshmemx_int16_and_reduce_block(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var);
  nvshmemx_int32_and_reduce_block(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var);
  nvshmemx_int64_and_reduce_block(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var);
  nvshmemx_uint8_and_reduce_block(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var);
  nvshmemx_uint16_and_reduce_block(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var);
  nvshmemx_uint32_and_reduce_block(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var);
  nvshmemx_uint64_and_reduce_block(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var);
  nvshmemx_size_and_reduce_block(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var);

  // nvshmemx_TYPENAME_and_reduce_warp (AOX)
  nvshmemx_uchar_and_reduce_warp(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var);
  nvshmemx_ushort_and_reduce_warp(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var);
  nvshmemx_uint_and_reduce_warp(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var);
  nvshmemx_ulong_and_reduce_warp(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var);
  nvshmemx_ulonglong_and_reduce_warp(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var);
  nvshmemx_int8_and_reduce_warp(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var);
  nvshmemx_int16_and_reduce_warp(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var);
  nvshmemx_int32_and_reduce_warp(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var);
  nvshmemx_int64_and_reduce_warp(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var);
  nvshmemx_uint8_and_reduce_warp(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var);
  nvshmemx_uint16_and_reduce_warp(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var);
  nvshmemx_uint32_and_reduce_warp(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var);
  nvshmemx_uint64_and_reduce_warp(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var);
  nvshmemx_size_and_reduce_warp(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var);

  // nvshmemx_TYPENAME_broadcast_block (RMA)
  nvshmemx_float_broadcast_block(nvshmem_team_t_var, &float_var, &const_float_var, size_var, int_var);
  nvshmemx_double_broadcast_block(nvshmem_team_t_var, &double_var, &const_double_var, size_var, int_var);
  nvshmemx_char_broadcast_block(nvshmem_team_t_var, &char_var, &const_char_var, size_var, int_var);
  nvshmemx_schar_broadcast_block(nvshmem_team_t_var, &schar_var, &const_signed_char_var, size_var, int_var);
  nvshmemx_short_broadcast_block(nvshmem_team_t_var, &short_var, &const_short_var, size_var, int_var);
  nvshmemx_int_broadcast_block(nvshmem_team_t_var, &int_var, &const_int_var, size_var, int_var);
  nvshmemx_long_broadcast_block(nvshmem_team_t_var, &long_var, &const_long_var, size_var, int_var);
  nvshmemx_longlong_broadcast_block(nvshmem_team_t_var, &longlong_var, &const_long_long_var, size_var, int_var);
  nvshmemx_uchar_broadcast_block(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var, int_var);
  nvshmemx_ushort_broadcast_block(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var, int_var);
  nvshmemx_uint_broadcast_block(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var, int_var);
  nvshmemx_ulong_broadcast_block(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var, int_var);
  nvshmemx_ulonglong_broadcast_block(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var, int_var);
  nvshmemx_int8_broadcast_block(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var, int_var);
  nvshmemx_int16_broadcast_block(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var, int_var);
  nvshmemx_int32_broadcast_block(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var, int_var);
  nvshmemx_int64_broadcast_block(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var, int_var);
  nvshmemx_uint8_broadcast_block(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var, int_var);
  nvshmemx_uint16_broadcast_block(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var, int_var);
  nvshmemx_uint32_broadcast_block(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var, int_var);
  nvshmemx_uint64_broadcast_block(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var, int_var);
  nvshmemx_size_broadcast_block(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var, int_var);
  nvshmemx_ptrdiff_broadcast_block(nvshmem_team_t_var, &ptrdiff_var, &const_ptrdiff_t_var, size_var, int_var);

  // nvshmemx_TYPENAME_broadcast_warp (RMA)
  nvshmemx_float_broadcast_warp(nvshmem_team_t_var, &float_var, &const_float_var, size_var, int_var);
  nvshmemx_double_broadcast_warp(nvshmem_team_t_var, &double_var, &const_double_var, size_var, int_var);
  nvshmemx_char_broadcast_warp(nvshmem_team_t_var, &char_var, &const_char_var, size_var, int_var);
  nvshmemx_schar_broadcast_warp(nvshmem_team_t_var, &schar_var, &const_signed_char_var, size_var, int_var);
  nvshmemx_short_broadcast_warp(nvshmem_team_t_var, &short_var, &const_short_var, size_var, int_var);
  nvshmemx_int_broadcast_warp(nvshmem_team_t_var, &int_var, &const_int_var, size_var, int_var);
  nvshmemx_long_broadcast_warp(nvshmem_team_t_var, &long_var, &const_long_var, size_var, int_var);
  nvshmemx_longlong_broadcast_warp(nvshmem_team_t_var, &longlong_var, &const_long_long_var, size_var, int_var);
  nvshmemx_uchar_broadcast_warp(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var, int_var);
  nvshmemx_ushort_broadcast_warp(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var, int_var);
  nvshmemx_uint_broadcast_warp(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var, int_var);
  nvshmemx_ulong_broadcast_warp(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var, int_var);
  nvshmemx_ulonglong_broadcast_warp(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var, int_var);
  nvshmemx_int8_broadcast_warp(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var, int_var);
  nvshmemx_int16_broadcast_warp(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var, int_var);
  nvshmemx_int32_broadcast_warp(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var, int_var);
  nvshmemx_int64_broadcast_warp(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var, int_var);
  nvshmemx_uint8_broadcast_warp(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var, int_var);
  nvshmemx_uint16_broadcast_warp(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var, int_var);
  nvshmemx_uint32_broadcast_warp(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var, int_var);
  nvshmemx_uint64_broadcast_warp(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var, int_var);
  nvshmemx_size_broadcast_warp(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var, int_var);
  nvshmemx_ptrdiff_broadcast_warp(nvshmem_team_t_var, &ptrdiff_var, &const_ptrdiff_t_var, size_var, int_var);

  // nvshmemx_TYPENAME_fcollect_block (RMA)
  nvshmemx_float_fcollect_block(nvshmem_team_t_var, &float_var, &const_float_var, size_var);
  nvshmemx_double_fcollect_block(nvshmem_team_t_var, &double_var, &const_double_var, size_var);
  nvshmemx_char_fcollect_block(nvshmem_team_t_var, &char_var, &const_char_var, size_var);
  nvshmemx_schar_fcollect_block(nvshmem_team_t_var, &schar_var, &const_signed_char_var, size_var);
  nvshmemx_short_fcollect_block(nvshmem_team_t_var, &short_var, &const_short_var, size_var);
  nvshmemx_int_fcollect_block(nvshmem_team_t_var, &int_var, &const_int_var, size_var);
  nvshmemx_long_fcollect_block(nvshmem_team_t_var, &long_var, &const_long_var, size_var);
  nvshmemx_longlong_fcollect_block(nvshmem_team_t_var, &longlong_var, &const_long_long_var, size_var);
  nvshmemx_uchar_fcollect_block(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var);
  nvshmemx_ushort_fcollect_block(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var);
  nvshmemx_uint_fcollect_block(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var);
  nvshmemx_ulong_fcollect_block(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var);
  nvshmemx_ulonglong_fcollect_block(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var);
  nvshmemx_int8_fcollect_block(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var);
  nvshmemx_int16_fcollect_block(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var);
  nvshmemx_int32_fcollect_block(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var);
  nvshmemx_int64_fcollect_block(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var);
  nvshmemx_uint8_fcollect_block(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var);
  nvshmemx_uint16_fcollect_block(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var);
  nvshmemx_uint32_fcollect_block(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var);
  nvshmemx_uint64_fcollect_block(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var);
  nvshmemx_size_fcollect_block(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var);
  nvshmemx_ptrdiff_fcollect_block(nvshmem_team_t_var, &ptrdiff_var, &const_ptrdiff_t_var, size_var);

  // nvshmemx_TYPENAME_fcollect_warp (RMA)
  nvshmemx_float_fcollect_warp(nvshmem_team_t_var, &float_var, &const_float_var, size_var);
  nvshmemx_double_fcollect_warp(nvshmem_team_t_var, &double_var, &const_double_var, size_var);
  nvshmemx_char_fcollect_warp(nvshmem_team_t_var, &char_var, &const_char_var, size_var);
  nvshmemx_schar_fcollect_warp(nvshmem_team_t_var, &schar_var, &const_signed_char_var, size_var);
  nvshmemx_short_fcollect_warp(nvshmem_team_t_var, &short_var, &const_short_var, size_var);
  nvshmemx_int_fcollect_warp(nvshmem_team_t_var, &int_var, &const_int_var, size_var);
  nvshmemx_long_fcollect_warp(nvshmem_team_t_var, &long_var, &const_long_var, size_var);
  nvshmemx_longlong_fcollect_warp(nvshmem_team_t_var, &longlong_var, &const_long_long_var, size_var);
  nvshmemx_uchar_fcollect_warp(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var);
  nvshmemx_ushort_fcollect_warp(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var);
  nvshmemx_uint_fcollect_warp(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var);
  nvshmemx_ulong_fcollect_warp(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var);
  nvshmemx_ulonglong_fcollect_warp(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var);
  nvshmemx_int8_fcollect_warp(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var);
  nvshmemx_int16_fcollect_warp(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var);
  nvshmemx_int32_fcollect_warp(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var);
  nvshmemx_int64_fcollect_warp(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var);
  nvshmemx_uint8_fcollect_warp(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var);
  nvshmemx_uint16_fcollect_warp(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var);
  nvshmemx_uint32_fcollect_warp(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var);
  nvshmemx_uint64_fcollect_warp(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var);
  nvshmemx_size_fcollect_warp(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var);
  nvshmemx_ptrdiff_fcollect_warp(nvshmem_team_t_var, &ptrdiff_var, &const_ptrdiff_t_var, size_var);

  // nvshmemx_TYPENAME_max_reduce_block (RMA)
  nvshmemx_float_max_reduce_block(nvshmem_team_t_var, &float_var, &const_float_var, size_var);
  nvshmemx_double_max_reduce_block(nvshmem_team_t_var, &double_var, &const_double_var, size_var);
  nvshmemx_char_max_reduce_block(nvshmem_team_t_var, &char_var, &const_char_var, size_var);
  nvshmemx_schar_max_reduce_block(nvshmem_team_t_var, &schar_var, &const_signed_char_var, size_var);
  nvshmemx_short_max_reduce_block(nvshmem_team_t_var, &short_var, &const_short_var, size_var);
  nvshmemx_int_max_reduce_block(nvshmem_team_t_var, &int_var, &const_int_var, size_var);
  nvshmemx_long_max_reduce_block(nvshmem_team_t_var, &long_var, &const_long_var, size_var);
  nvshmemx_longlong_max_reduce_block(nvshmem_team_t_var, &longlong_var, &const_long_long_var, size_var);
  nvshmemx_uchar_max_reduce_block(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var);
  nvshmemx_ushort_max_reduce_block(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var);
  nvshmemx_uint_max_reduce_block(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var);
  nvshmemx_ulong_max_reduce_block(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var);
  nvshmemx_ulonglong_max_reduce_block(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var);
  nvshmemx_int8_max_reduce_block(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var);
  nvshmemx_int16_max_reduce_block(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var);
  nvshmemx_int32_max_reduce_block(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var);
  nvshmemx_int64_max_reduce_block(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var);
  nvshmemx_uint8_max_reduce_block(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var);
  nvshmemx_uint16_max_reduce_block(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var);
  nvshmemx_uint32_max_reduce_block(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var);
  nvshmemx_uint64_max_reduce_block(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var);
  nvshmemx_size_max_reduce_block(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var);
  // nvshmemx_ptrdiff_max_reduce_block(nvshmem_team_t_var, &ptrdiff_var, &const_ptrdiff_t_var, size_var);

  // nvshmemx_TYPENAME_max_reduce_warp (RMA)
  nvshmemx_float_max_reduce_warp(nvshmem_team_t_var, &float_var, &const_float_var, size_var);
  nvshmemx_double_max_reduce_warp(nvshmem_team_t_var, &double_var, &const_double_var, size_var);
  nvshmemx_char_max_reduce_warp(nvshmem_team_t_var, &char_var, &const_char_var, size_var);
  nvshmemx_schar_max_reduce_warp(nvshmem_team_t_var, &schar_var, &const_signed_char_var, size_var);
  nvshmemx_short_max_reduce_warp(nvshmem_team_t_var, &short_var, &const_short_var, size_var);
  nvshmemx_int_max_reduce_warp(nvshmem_team_t_var, &int_var, &const_int_var, size_var);
  nvshmemx_long_max_reduce_warp(nvshmem_team_t_var, &long_var, &const_long_var, size_var);
  nvshmemx_longlong_max_reduce_warp(nvshmem_team_t_var, &longlong_var, &const_long_long_var, size_var);
  nvshmemx_uchar_max_reduce_warp(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var);
  nvshmemx_ushort_max_reduce_warp(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var);
  nvshmemx_uint_max_reduce_warp(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var);
  nvshmemx_ulong_max_reduce_warp(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var);
  nvshmemx_ulonglong_max_reduce_warp(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var);
  nvshmemx_int8_max_reduce_warp(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var);
  nvshmemx_int16_max_reduce_warp(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var);
  nvshmemx_int32_max_reduce_warp(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var);
  nvshmemx_int64_max_reduce_warp(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var);
  nvshmemx_uint8_max_reduce_warp(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var);
  nvshmemx_uint16_max_reduce_warp(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var);
  nvshmemx_uint32_max_reduce_warp(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var);
  nvshmemx_uint64_max_reduce_warp(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var);
  nvshmemx_size_max_reduce_warp(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var);
  // nvshmemx_ptrdiff_max_reduce_warp(nvshmem_team_t_var, &ptrdiff_var, &const_ptrdiff_t_var, size_var);

  // nvshmemx_TYPENAME_min_reduce_block (RMA)
  nvshmemx_float_min_reduce_block(nvshmem_team_t_var, &float_var, &const_float_var, size_var);
  nvshmemx_double_min_reduce_block(nvshmem_team_t_var, &double_var, &const_double_var, size_var);
  nvshmemx_char_min_reduce_block(nvshmem_team_t_var, &char_var, &const_char_var, size_var);
  nvshmemx_schar_min_reduce_block(nvshmem_team_t_var, &schar_var, &const_signed_char_var, size_var);
  nvshmemx_short_min_reduce_block(nvshmem_team_t_var, &short_var, &const_short_var, size_var);
  nvshmemx_int_min_reduce_block(nvshmem_team_t_var, &int_var, &const_int_var, size_var);
  nvshmemx_long_min_reduce_block(nvshmem_team_t_var, &long_var, &const_long_var, size_var);
  nvshmemx_longlong_min_reduce_block(nvshmem_team_t_var, &longlong_var, &const_long_long_var, size_var);
  nvshmemx_uchar_min_reduce_block(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var);
  nvshmemx_ushort_min_reduce_block(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var);
  nvshmemx_uint_min_reduce_block(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var);
  nvshmemx_ulong_min_reduce_block(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var);
  nvshmemx_ulonglong_min_reduce_block(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var);
  nvshmemx_int8_min_reduce_block(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var);
  nvshmemx_int16_min_reduce_block(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var);
  nvshmemx_int32_min_reduce_block(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var);
  nvshmemx_int64_min_reduce_block(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var);
  nvshmemx_uint8_min_reduce_block(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var);
  nvshmemx_uint16_min_reduce_block(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var);
  nvshmemx_uint32_min_reduce_block(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var);
  nvshmemx_uint64_min_reduce_block(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var);
  nvshmemx_size_min_reduce_block(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var);
  // nvshmemx_ptrdiff_min_reduce_block(nvshmem_team_t_var, &ptrdiff_var, &const_ptrdiff_t_var, size_var);

  // nvshmemx_TYPENAME_min_reduce_warp (RMA)
  nvshmemx_float_min_reduce_warp(nvshmem_team_t_var, &float_var, &const_float_var, size_var);
  nvshmemx_double_min_reduce_warp(nvshmem_team_t_var, &double_var, &const_double_var, size_var);
  nvshmemx_char_min_reduce_warp(nvshmem_team_t_var, &char_var, &const_char_var, size_var);
  nvshmemx_schar_min_reduce_warp(nvshmem_team_t_var, &schar_var, &const_signed_char_var, size_var);
  nvshmemx_short_min_reduce_warp(nvshmem_team_t_var, &short_var, &const_short_var, size_var);
  nvshmemx_int_min_reduce_warp(nvshmem_team_t_var, &int_var, &const_int_var, size_var);
  nvshmemx_long_min_reduce_warp(nvshmem_team_t_var, &long_var, &const_long_var, size_var);
  nvshmemx_longlong_min_reduce_warp(nvshmem_team_t_var, &longlong_var, &const_long_long_var, size_var);
  nvshmemx_uchar_min_reduce_warp(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var);
  nvshmemx_ushort_min_reduce_warp(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var);
  nvshmemx_uint_min_reduce_warp(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var);
  nvshmemx_ulong_min_reduce_warp(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var);
  nvshmemx_ulonglong_min_reduce_warp(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var);
  nvshmemx_int8_min_reduce_warp(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var);
  nvshmemx_int16_min_reduce_warp(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var);
  nvshmemx_int32_min_reduce_warp(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var);
  nvshmemx_int64_min_reduce_warp(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var);
  nvshmemx_uint8_min_reduce_warp(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var);
  nvshmemx_uint16_min_reduce_warp(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var);
  nvshmemx_uint32_min_reduce_warp(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var);
  nvshmemx_uint64_min_reduce_warp(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var);
  nvshmemx_size_min_reduce_warp(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var);
  // nvshmemx_ptrdiff_min_reduce_warp(nvshmem_team_t_var, &ptrdiff_var, &const_ptrdiff_t_var, size_var);

  // nvshmemx_TYPENAME_or_reduce_block (AOX)
  nvshmemx_uchar_or_reduce_block(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var);
  nvshmemx_ushort_or_reduce_block(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var);
  nvshmemx_uint_or_reduce_block(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var);
  nvshmemx_ulong_or_reduce_block(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var);
  nvshmemx_ulonglong_or_reduce_block(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var);
  nvshmemx_int8_or_reduce_block(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var);
  nvshmemx_int16_or_reduce_block(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var);
  nvshmemx_int32_or_reduce_block(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var);
  nvshmemx_int64_or_reduce_block(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var);
  nvshmemx_uint8_or_reduce_block(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var);
  nvshmemx_uint16_or_reduce_block(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var);
  nvshmemx_uint32_or_reduce_block(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var);
  nvshmemx_uint64_or_reduce_block(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var);
  nvshmemx_size_or_reduce_block(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var);

  // nvshmemx_TYPENAME_or_reduce_warp (AOX)
  nvshmemx_uchar_or_reduce_warp(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var);
  nvshmemx_ushort_or_reduce_warp(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var);
  nvshmemx_uint_or_reduce_warp(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var);
  nvshmemx_ulong_or_reduce_warp(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var);
  nvshmemx_ulonglong_or_reduce_warp(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var);
  nvshmemx_int8_or_reduce_warp(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var);
  nvshmemx_int16_or_reduce_warp(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var);
  nvshmemx_int32_or_reduce_warp(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var);
  nvshmemx_int64_or_reduce_warp(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var);
  nvshmemx_uint8_or_reduce_warp(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var);
  nvshmemx_uint16_or_reduce_warp(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var);
  nvshmemx_uint32_or_reduce_warp(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var);
  nvshmemx_uint64_or_reduce_warp(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var);
  nvshmemx_size_or_reduce_warp(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var);

  // nvshmemx_TYPENAME_prod_reduce_block (RMA)
  nvshmemx_float_prod_reduce_block(nvshmem_team_t_var, &float_var, &const_float_var, size_var);
  nvshmemx_double_prod_reduce_block(nvshmem_team_t_var, &double_var, &const_double_var, size_var);
  nvshmemx_char_prod_reduce_block(nvshmem_team_t_var, &char_var, &const_char_var, size_var);
  nvshmemx_schar_prod_reduce_block(nvshmem_team_t_var, &schar_var, &const_signed_char_var, size_var);
  nvshmemx_short_prod_reduce_block(nvshmem_team_t_var, &short_var, &const_short_var, size_var);
  nvshmemx_int_prod_reduce_block(nvshmem_team_t_var, &int_var, &const_int_var, size_var);
  nvshmemx_long_prod_reduce_block(nvshmem_team_t_var, &long_var, &const_long_var, size_var);
  nvshmemx_longlong_prod_reduce_block(nvshmem_team_t_var, &longlong_var, &const_long_long_var, size_var);
  nvshmemx_uchar_prod_reduce_block(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var);
  nvshmemx_ushort_prod_reduce_block(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var);
  nvshmemx_uint_prod_reduce_block(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var);
  nvshmemx_ulong_prod_reduce_block(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var);
  nvshmemx_ulonglong_prod_reduce_block(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var);
  nvshmemx_int8_prod_reduce_block(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var);
  nvshmemx_int16_prod_reduce_block(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var);
  nvshmemx_int32_prod_reduce_block(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var);
  nvshmemx_int64_prod_reduce_block(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var);
  nvshmemx_uint8_prod_reduce_block(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var);
  nvshmemx_uint16_prod_reduce_block(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var);
  nvshmemx_uint32_prod_reduce_block(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var);
  nvshmemx_uint64_prod_reduce_block(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var);
  nvshmemx_size_prod_reduce_block(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var);
  // nvshmemx_ptrdiff_prod_reduce_block(nvshmem_team_t_var, &ptrdiff_var, &const_ptrdiff_t_var, size_var);

  // nvshmemx_TYPENAME_prod_reduce_warp (RMA)
  nvshmemx_float_prod_reduce_warp(nvshmem_team_t_var, &float_var, &const_float_var, size_var);
  nvshmemx_double_prod_reduce_warp(nvshmem_team_t_var, &double_var, &const_double_var, size_var);
  nvshmemx_char_prod_reduce_warp(nvshmem_team_t_var, &char_var, &const_char_var, size_var);
  nvshmemx_schar_prod_reduce_warp(nvshmem_team_t_var, &schar_var, &const_signed_char_var, size_var);
  nvshmemx_short_prod_reduce_warp(nvshmem_team_t_var, &short_var, &const_short_var, size_var);
  nvshmemx_int_prod_reduce_warp(nvshmem_team_t_var, &int_var, &const_int_var, size_var);
  nvshmemx_long_prod_reduce_warp(nvshmem_team_t_var, &long_var, &const_long_var, size_var);
  nvshmemx_longlong_prod_reduce_warp(nvshmem_team_t_var, &longlong_var, &const_long_long_var, size_var);
  nvshmemx_uchar_prod_reduce_warp(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var);
  nvshmemx_ushort_prod_reduce_warp(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var);
  nvshmemx_uint_prod_reduce_warp(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var);
  nvshmemx_ulong_prod_reduce_warp(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var);
  nvshmemx_ulonglong_prod_reduce_warp(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var);
  nvshmemx_int8_prod_reduce_warp(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var);
  nvshmemx_int16_prod_reduce_warp(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var);
  nvshmemx_int32_prod_reduce_warp(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var);
  nvshmemx_int64_prod_reduce_warp(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var);
  nvshmemx_uint8_prod_reduce_warp(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var);
  nvshmemx_uint16_prod_reduce_warp(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var);
  nvshmemx_uint32_prod_reduce_warp(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var);
  nvshmemx_uint64_prod_reduce_warp(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var);
  nvshmemx_size_prod_reduce_warp(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var);
  // nvshmemx_ptrdiff_prod_reduce_warp(nvshmem_team_t_var, &ptrdiff_var, &const_ptrdiff_t_var, size_var);

  // nvshmemx_TYPENAME_sum_reduce_block (RMA)
  nvshmemx_float_sum_reduce_block(nvshmem_team_t_var, &float_var, &const_float_var, size_var);
  nvshmemx_double_sum_reduce_block(nvshmem_team_t_var, &double_var, &const_double_var, size_var);
  nvshmemx_char_sum_reduce_block(nvshmem_team_t_var, &char_var, &const_char_var, size_var);
  nvshmemx_schar_sum_reduce_block(nvshmem_team_t_var, &schar_var, &const_signed_char_var, size_var);
  nvshmemx_short_sum_reduce_block(nvshmem_team_t_var, &short_var, &const_short_var, size_var);
  nvshmemx_int_sum_reduce_block(nvshmem_team_t_var, &int_var, &const_int_var, size_var);
  nvshmemx_long_sum_reduce_block(nvshmem_team_t_var, &long_var, &const_long_var, size_var);
  nvshmemx_longlong_sum_reduce_block(nvshmem_team_t_var, &longlong_var, &const_long_long_var, size_var);
  nvshmemx_uchar_sum_reduce_block(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var);
  nvshmemx_ushort_sum_reduce_block(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var);
  nvshmemx_uint_sum_reduce_block(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var);
  nvshmemx_ulong_sum_reduce_block(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var);
  nvshmemx_ulonglong_sum_reduce_block(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var);
  nvshmemx_int8_sum_reduce_block(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var);
  nvshmemx_int16_sum_reduce_block(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var);
  nvshmemx_int32_sum_reduce_block(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var);
  nvshmemx_int64_sum_reduce_block(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var);
  nvshmemx_uint8_sum_reduce_block(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var);
  nvshmemx_uint16_sum_reduce_block(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var);
  nvshmemx_uint32_sum_reduce_block(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var);
  nvshmemx_uint64_sum_reduce_block(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var);
  nvshmemx_size_sum_reduce_block(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var);
  // nvshmemx_ptrdiff_sum_reduce_block(nvshmem_team_t_var, &ptrdiff_var, &const_ptrdiff_t_var, size_var);

  // nvshmemx_TYPENAME_sum_reduce_warp (RMA)
  nvshmemx_float_sum_reduce_warp(nvshmem_team_t_var, &float_var, &const_float_var, size_var);
  nvshmemx_double_sum_reduce_warp(nvshmem_team_t_var, &double_var, &const_double_var, size_var);
  nvshmemx_char_sum_reduce_warp(nvshmem_team_t_var, &char_var, &const_char_var, size_var);
  nvshmemx_schar_sum_reduce_warp(nvshmem_team_t_var, &schar_var, &const_signed_char_var, size_var);
  nvshmemx_short_sum_reduce_warp(nvshmem_team_t_var, &short_var, &const_short_var, size_var);
  nvshmemx_int_sum_reduce_warp(nvshmem_team_t_var, &int_var, &const_int_var, size_var);
  nvshmemx_long_sum_reduce_warp(nvshmem_team_t_var, &long_var, &const_long_var, size_var);
  nvshmemx_longlong_sum_reduce_warp(nvshmem_team_t_var, &longlong_var, &const_long_long_var, size_var);
  nvshmemx_uchar_sum_reduce_warp(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var);
  nvshmemx_ushort_sum_reduce_warp(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var);
  nvshmemx_uint_sum_reduce_warp(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var);
  nvshmemx_ulong_sum_reduce_warp(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var);
  nvshmemx_ulonglong_sum_reduce_warp(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var);
  nvshmemx_int8_sum_reduce_warp(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var);
  nvshmemx_int16_sum_reduce_warp(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var);
  nvshmemx_int32_sum_reduce_warp(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var);
  nvshmemx_int64_sum_reduce_warp(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var);
  nvshmemx_uint8_sum_reduce_warp(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var);
  nvshmemx_uint16_sum_reduce_warp(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var);
  nvshmemx_uint32_sum_reduce_warp(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var);
  nvshmemx_uint64_sum_reduce_warp(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var);
  nvshmemx_size_sum_reduce_warp(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var);
  // nvshmemx_ptrdiff_sum_reduce_warp(nvshmem_team_t_var, &ptrdiff_var, &const_ptrdiff_t_var, size_var);

  // nvshmemx_TYPENAME_xor_reduce_block (AOX)
  nvshmemx_uchar_xor_reduce_block(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var);
  nvshmemx_ushort_xor_reduce_block(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var);
  nvshmemx_uint_xor_reduce_block(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var);
  nvshmemx_ulong_xor_reduce_block(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var);
  nvshmemx_ulonglong_xor_reduce_block(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var);
  nvshmemx_int8_xor_reduce_block(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var);
  nvshmemx_int16_xor_reduce_block(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var);
  nvshmemx_int32_xor_reduce_block(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var);
  nvshmemx_int64_xor_reduce_block(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var);
  nvshmemx_uint8_xor_reduce_block(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var);
  nvshmemx_uint16_xor_reduce_block(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var);
  nvshmemx_uint32_xor_reduce_block(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var);
  nvshmemx_uint64_xor_reduce_block(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var);
  nvshmemx_size_xor_reduce_block(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var);

  // nvshmemx_TYPENAME_xor_reduce_warp (AOX)
  nvshmemx_uchar_xor_reduce_warp(nvshmem_team_t_var, &uchar_var, &const_unsigned_char_var, size_var);
  nvshmemx_ushort_xor_reduce_warp(nvshmem_team_t_var, &ushort_var, &const_unsigned_short_var, size_var);
  nvshmemx_uint_xor_reduce_warp(nvshmem_team_t_var, &uint_var, &const_unsigned_int_var, size_var);
  nvshmemx_ulong_xor_reduce_warp(nvshmem_team_t_var, &ulong_var, &const_unsigned_long_var, size_var);
  nvshmemx_ulonglong_xor_reduce_warp(nvshmem_team_t_var, &ulonglong_var, &const_unsigned_long_long_var, size_var);
  nvshmemx_int8_xor_reduce_warp(nvshmem_team_t_var, &int8_var, &const_int8_t_var, size_var);
  nvshmemx_int16_xor_reduce_warp(nvshmem_team_t_var, &int16_var, &const_int16_t_var, size_var);
  nvshmemx_int32_xor_reduce_warp(nvshmem_team_t_var, &int32_var, &const_int32_t_var, size_var);
  nvshmemx_int64_xor_reduce_warp(nvshmem_team_t_var, &int64_var, &const_int64_t_var, size_var);
  nvshmemx_uint8_xor_reduce_warp(nvshmem_team_t_var, &uint8_var, &const_uint8_t_var, size_var);
  nvshmemx_uint16_xor_reduce_warp(nvshmem_team_t_var, &uint16_var, &const_uint16_t_var, size_var);
  nvshmemx_uint32_xor_reduce_warp(nvshmem_team_t_var, &uint32_var, &const_uint32_t_var, size_var);
  nvshmemx_uint64_xor_reduce_warp(nvshmem_team_t_var, &uint64_var, &const_uint64_t_var, size_var);
  nvshmemx_size_xor_reduce_warp(nvshmem_team_t_var, &size_var, &const_size_t_var, size_var);

  nvshmemx_alltoallmem_block(nvshmem_team_t_var, void_ptr_var, const_void_var, size_var);
  nvshmemx_alltoallmem_warp(nvshmem_team_t_var, void_ptr_var, const_void_var, size_var);

  // nvshmemx_broadcastmem_block(nvshmem_team_t_var, void_ptr_var, const_void_var, size_var, int_var);
  // nvshmemx_broadcastmem_warp(nvshmem_team_t_var, void_ptr_var, const_void_var, size_var, int_var);

  // nvshmemx_fcollectmem_block(nvshmem_team_t_var, void_ptr_var, const_void_var, size_var);
  // nvshmemx_fcollectmem_warp(nvshmem_team_t_var, void_ptr_var, const_void_var, size_var);

  // nvshmemx_sync_block(nvshmem_team_t_var);
  // nvshmemx_sync_on_stream(nvshmem_team_t_var, cudaStream_t_var);
  // nvshmemx_sync_warp(nvshmem_team_t_var);

  nvshmemx_team_sync_block(nvshmem_team_t_var);
  nvshmemx_team_sync_warp(nvshmem_team_t_var);

  nvshmem_short_test_any(&short_var, size_var, &const_int_var, int_var, short_var);
  nvshmem_int_test_any(&int_var, size_var, &const_int_var, int_var, int_var);
  nvshmem_long_test_any(&long_var, size_var, &const_int_var, int_var, long_var);
  nvshmem_longlong_test_any(&longlong_var, size_var, &const_int_var, int_var, longlong_var);
  nvshmem_ushort_test_any(&ushort_var, size_var, &const_int_var, int_var, ushort_var);
  nvshmem_uint_test_any(&uint_var, size_var, &const_int_var, int_var, uint_var);
  nvshmem_ulong_test_any(&ulong_var, size_var, &const_int_var, int_var, ulong_var);
  nvshmem_ulonglong_test_any(&ulonglong_var, size_var, &const_int_var, int_var, ulonglong_var);
  nvshmem_int32_test_any(&int32_var, size_var, &const_int_var, int_var, int32_var);
  nvshmem_int64_test_any(&int64_var, size_var, &const_int_var, int_var, int64_var);
  nvshmem_uint32_test_any(&uint32_var, size_var, &const_int_var, int_var, uint32_var);
  nvshmem_uint64_test_any(&uint64_var, size_var, &const_int_var, int_var, uint64_var);
  nvshmem_size_test_any(&size_var, size_var, &const_int_var, int_var, size_var);
  nvshmem_ptrdiff_test_any(&ptrdiff_var, size_var, &const_int_var, int_var, ptrdiff_var);

  nvshmem_short_test_any_vector(&short_var, size_var, &const_int_var, int_var, &short_var);
  nvshmem_int_test_any_vector(&int_var, size_var, &const_int_var, int_var, &int_var);
  nvshmem_long_test_any_vector(&long_var, size_var, &const_int_var, int_var, &long_var);
  nvshmem_longlong_test_any_vector(&longlong_var, size_var, &const_int_var, int_var, &longlong_var);
  nvshmem_ushort_test_any_vector(&ushort_var, size_var, &const_int_var, int_var, &ushort_var);
  nvshmem_uint_test_any_vector(&uint_var, size_var, &const_int_var, int_var, &uint_var);
  nvshmem_ulong_test_any_vector(&ulong_var, size_var, &const_int_var, int_var, &ulong_var);
  nvshmem_ulonglong_test_any_vector(&ulonglong_var, size_var, &const_int_var, int_var, &ulonglong_var);
  nvshmem_int32_test_any_vector(&int32_var, size_var, &const_int_var, int_var, &int32_var);
  nvshmem_int64_test_any_vector(&int64_var, size_var, &const_int_var, int_var, &int64_var);
  nvshmem_uint32_test_any_vector(&uint32_var, size_var, &const_int_var, int_var, &uint32_var);
  nvshmem_uint64_test_any_vector(&uint64_var, size_var, &const_int_var, int_var, &uint64_var);
  nvshmem_size_test_any_vector(&size_var, size_var, &const_int_var, int_var, &size_var);
  nvshmem_ptrdiff_test_any_vector(&ptrdiff_var, size_var, &const_int_var, int_var, &ptrdiff_var);

  nvshmem_short_test_some(&short_var, size_var, &size_var, &const_int_var, int_var, short_var);
  nvshmem_int_test_some(&int_var, size_var, &size_var, &const_int_var, int_var, int_var);
  nvshmem_long_test_some(&long_var, size_var, &size_var, &const_int_var, int_var, long_var);
  nvshmem_longlong_test_some(&longlong_var, size_var, &size_var, &const_int_var, int_var, longlong_var);
  nvshmem_ushort_test_some(&ushort_var, size_var, &size_var, &const_int_var, int_var, ushort_var);
  nvshmem_uint_test_some(&uint_var, size_var, &size_var, &const_int_var, int_var, uint_var);
  nvshmem_ulong_test_some(&ulong_var, size_var, &size_var, &const_int_var, int_var, ulong_var);
  nvshmem_ulonglong_test_some(&ulonglong_var, size_var, &size_var, &const_int_var, int_var, ulonglong_var);
  nvshmem_int32_test_some(&int32_var, size_var, &size_var, &const_int_var, int_var, int32_var);
  nvshmem_int64_test_some(&int64_var, size_var, &size_var, &const_int_var, int_var, int64_var);
  nvshmem_uint32_test_some(&uint32_var, size_var, &size_var, &const_int_var, int_var, uint32_var);
  nvshmem_uint64_test_some(&uint64_var, size_var, &size_var, &const_int_var, int_var, uint64_var);
  nvshmem_size_test_some(&size_var, size_var, &size_var, &const_int_var, int_var, size_var);
  nvshmem_ptrdiff_test_some(&ptrdiff_var, size_var, &size_var, &const_int_var, int_var, ptrdiff_var);
  nvshmem_short_test_some_vector(&short_var, size_var, &size_var, &const_int_var, int_var, &short_var);
  nvshmem_int_test_some_vector(&int_var, size_var, &size_var, &const_int_var, int_var, &int_var);
  nvshmem_long_test_some_vector(&long_var, size_var, &size_var, &const_int_var, int_var, &long_var);
  nvshmem_longlong_test_some_vector(&longlong_var, size_var, &size_var, &const_int_var, int_var, &longlong_var);
  nvshmem_ushort_test_some_vector(&ushort_var, size_var, &size_var, &const_int_var, int_var, &ushort_var);
  nvshmem_uint_test_some_vector(&uint_var, size_var, &size_var, &const_int_var, int_var, &uint_var);
  nvshmem_ulong_test_some_vector(&ulong_var, size_var, &size_var, &const_int_var, int_var, &ulong_var);
  nvshmem_ulonglong_test_some_vector(&ulonglong_var, size_var, &size_var, &const_int_var, int_var, &ulonglong_var);
  nvshmem_int32_test_some_vector(&int32_var, size_var, &size_var, &const_int_var, int_var, &int32_var);
  nvshmem_int64_test_some_vector(&int64_var, size_var, &size_var, &const_int_var, int_var, &int64_var);
  nvshmem_uint32_test_some_vector(&uint32_var, size_var, &size_var, &const_int_var, int_var, &uint32_var);
  nvshmem_uint64_test_some_vector(&uint64_var, size_var, &size_var, &const_int_var, int_var, &uint64_var);
  nvshmem_size_test_some_vector(&size_var, size_var, &size_var, &const_int_var, int_var, &size_var);
  nvshmem_ptrdiff_test_some_vector(&ptrdiff_var, size_var, &size_var, &const_int_var, int_var, &ptrdiff_var);

  nvshmem_short_wait_until_any(&short_var, size_var, &const_int_var, int_var, short_var);
  nvshmem_int_wait_until_any(&int_var, size_var, &const_int_var, int_var, int_var);
  nvshmem_long_wait_until_any(&long_var, size_var, &const_int_var, int_var, long_var);
  nvshmem_longlong_wait_until_any(&longlong_var, size_var, &const_int_var, int_var, longlong_var);
  nvshmem_ushort_wait_until_any(&ushort_var, size_var, &const_int_var, int_var, ushort_var);
  nvshmem_uint_wait_until_any(&uint_var, size_var, &const_int_var, int_var, uint_var);
  nvshmem_ulong_wait_until_any(&ulong_var, size_var, &const_int_var, int_var, ulong_var);
  nvshmem_ulonglong_wait_until_any(&ulonglong_var, size_var, &const_int_var, int_var, ulonglong_var);
  nvshmem_int32_wait_until_any(&int32_var, size_var, &const_int_var, int_var, int32_var);
  nvshmem_int64_wait_until_any(&int64_var, size_var, &const_int_var, int_var, int64_var);
  nvshmem_uint32_wait_until_any(&uint32_var, size_var, &const_int_var, int_var, uint32_var);
  nvshmem_uint64_wait_until_any(&uint64_var, size_var, &const_int_var, int_var, uint64_var);
  nvshmem_size_wait_until_any(&size_var, size_var, &const_int_var, int_var, size_var);
  nvshmem_ptrdiff_wait_until_any(&ptrdiff_var, size_var, &const_int_var, int_var, ptrdiff_var);

  nvshmem_short_wait_until_any_vector(&short_var, size_var, &const_int_var, int_var, &short_var);
  nvshmem_int_wait_until_any_vector(&int_var, size_var, &const_int_var, int_var, &int_var);
  nvshmem_long_wait_until_any_vector(&long_var, size_var, &const_int_var, int_var, &long_var);
  nvshmem_longlong_wait_until_any_vector(&longlong_var, size_var, &const_int_var, int_var, &longlong_var);
  nvshmem_ushort_wait_until_any_vector(&ushort_var, size_var, &const_int_var, int_var, &ushort_var);
  nvshmem_uint_wait_until_any_vector(&uint_var, size_var, &const_int_var, int_var, &uint_var);
  nvshmem_ulong_wait_until_any_vector(&ulong_var, size_var, &const_int_var, int_var, &ulong_var);
  nvshmem_ulonglong_wait_until_any_vector(&ulonglong_var, size_var, &const_int_var, int_var, &ulonglong_var);
  nvshmem_int32_wait_until_any_vector(&int32_var, size_var, &const_int_var, int_var, &int32_var);
  nvshmem_int64_wait_until_any_vector(&int64_var, size_var, &const_int_var, int_var, &int64_var);
  nvshmem_uint32_wait_until_any_vector(&uint32_var, size_var, &const_int_var, int_var, &uint32_var);
  nvshmem_uint64_wait_until_any_vector(&uint64_var, size_var, &const_int_var, int_var, &uint64_var);
  nvshmem_size_wait_until_any_vector(&size_var, size_var, &const_int_var, int_var, &size_var);
  nvshmem_ptrdiff_wait_until_any_vector(&ptrdiff_var, size_var, &const_int_var, int_var, &ptrdiff_var);

  nvshmem_short_wait_until_some(&short_var, size_var, &size_var, &const_int_var, int_var, short_var);
  nvshmem_int_wait_until_some(&int_var, size_var, &size_var, &const_int_var, int_var, int_var);
  nvshmem_long_wait_until_some(&long_var, size_var, &size_var, &const_int_var, int_var, long_var);
  nvshmem_longlong_wait_until_some(&longlong_var, size_var, &size_var, &const_int_var, int_var, longlong_var);
  nvshmem_ushort_wait_until_some(&ushort_var, size_var, &size_var, &const_int_var, int_var, ushort_var);
  nvshmem_uint_wait_until_some(&uint_var, size_var, &size_var, &const_int_var, int_var, uint_var);
  nvshmem_ulong_wait_until_some(&ulong_var, size_var, &size_var, &const_int_var, int_var, ulong_var);
  nvshmem_ulonglong_wait_until_some(&ulonglong_var, size_var, &size_var, &const_int_var, int_var, ulonglong_var);
  nvshmem_int32_wait_until_some(&int32_var, size_var, &size_var, &const_int_var, int_var, int32_var);
  nvshmem_int64_wait_until_some(&int64_var, size_var, &size_var, &const_int_var, int_var, int64_var);
  nvshmem_uint32_wait_until_some(&uint32_var, size_var, &size_var, &const_int_var, int_var, uint32_var);
  nvshmem_uint64_wait_until_some(&uint64_var, size_var, &size_var, &const_int_var, int_var, uint64_var);
  nvshmem_size_wait_until_some(&size_var, size_var, &size_var, &const_int_var, int_var, size_var);
  nvshmem_ptrdiff_wait_until_some(&ptrdiff_var, size_var, &size_var, &const_int_var, int_var, ptrdiff_var);

  nvshmem_short_wait_until_some_vector(&short_var, size_var, &size_var, &const_int_var, int_var, &short_var);
  nvshmem_int_wait_until_some_vector(&int_var, size_var, &size_var, &const_int_var, int_var, &int_var);
  nvshmem_long_wait_until_some_vector(&long_var, size_var, &size_var, &const_int_var, int_var, &long_var);
  nvshmem_longlong_wait_until_some_vector(&longlong_var, size_var, &size_var, &const_int_var, int_var, &longlong_var);
  nvshmem_ushort_wait_until_some_vector(&ushort_var, size_var, &size_var, &const_int_var, int_var, &ushort_var);
  nvshmem_uint_wait_until_some_vector(&uint_var, size_var, &size_var, &const_int_var, int_var, &uint_var);
  nvshmem_ulong_wait_until_some_vector(&ulong_var, size_var, &size_var, &const_int_var, int_var, &ulong_var);
  nvshmem_ulonglong_wait_until_some_vector(&ulonglong_var, size_var, &size_var, &const_int_var, int_var, &ulonglong_var);
  nvshmem_int32_wait_until_some_vector(&int32_var, size_var, &size_var, &const_int_var, int_var, &int32_var);
  nvshmem_int64_wait_until_some_vector(&int64_var, size_var, &size_var, &const_int_var, int_var, &int64_var);
  nvshmem_uint32_wait_until_some_vector(&uint32_var, size_var, &size_var, &const_int_var, int_var, &uint32_var);
  nvshmem_uint64_wait_until_some_vector(&uint64_var, size_var, &size_var, &const_int_var, int_var, &uint64_var);
  nvshmem_size_wait_until_some_vector(&size_var, size_var, &size_var, &const_int_var, int_var, &size_var);
  nvshmem_ptrdiff_wait_until_some_vector(&ptrdiff_var, size_var, &size_var, &const_int_var, int_var, &ptrdiff_var);

  nvshmem_signal_fetch(&uint64_var);
  nvshmem_signal_wait_until(&uint64_var, int_var, uint64_var);

  nvshmem_int_atomic_add(&int_var, int_var, int_var);
  nvshmem_long_atomic_add(&long_var, long_var, int_var);
  nvshmem_longlong_atomic_add(&longlong_var, longlong_var, int_var);
  nvshmem_size_atomic_add(&size_var, size_var, int_var);
  nvshmem_ptrdiff_atomic_add(&ptrdiff_var, ptrdiff_var, int_var);
  nvshmem_uint_atomic_and(&uint_var, uint_var, int_var);
  nvshmem_ulong_atomic_and(&ulong_var, ulong_var, int_var);
  nvshmem_ulonglong_atomic_and(&ulonglong_var, ulonglong_var, int_var);
  nvshmem_int32_atomic_and(&int32_var, int32_var, int_var);
  nvshmem_int64_atomic_and(&int64_var, int64_var, int_var);
  nvshmem_uint32_atomic_and(&uint32_var, uint32_var, int_var);
  nvshmem_uint64_atomic_and(&uint64_var, uint64_var, int_var);

  nvshmem_int_atomic_inc(&int_var, int_var);
  nvshmem_long_atomic_inc(&long_var, int_var);
  nvshmem_longlong_atomic_inc(&longlong_var, int_var);
  nvshmem_size_atomic_inc(&size_var, int_var);
  nvshmem_ptrdiff_atomic_inc(&ptrdiff_var, int_var);

  nvshmem_uint_atomic_or(&uint_var, uint_var, int_var);
  nvshmem_ulong_atomic_or(&ulong_var, ulong_var, int_var);
  nvshmem_ulonglong_atomic_or(&ulonglong_var, ulonglong_var, int_var);
  nvshmem_int32_atomic_or(&int32_var, int32_var, int_var);
  nvshmem_int64_atomic_or(&int64_var, int64_var, int_var);
  nvshmem_uint32_atomic_or(&uint32_var, uint32_var, int_var);
  nvshmem_uint64_atomic_or(&uint64_var, uint64_var, int_var);

  nvshmem_float_atomic_set(&float_var, float_var, int_var);
  nvshmem_double_atomic_set(&double_var, double_var, int_var);
  nvshmem_int_atomic_set(&int_var, int_var, int_var);
  nvshmem_long_atomic_set(&long_var, long_var, int_var);
  nvshmem_longlong_atomic_set(&longlong_var, longlong_var, int_var);
  nvshmem_size_atomic_set(&size_var, size_var, int_var);
  nvshmem_ptrdiff_atomic_set(&ptrdiff_var, ptrdiff_var, int_var);

  nvshmem_uint_atomic_xor(&uint_var, uint_var, int_var);
  nvshmem_ulong_atomic_xor(&ulong_var, ulong_var, int_var);
  nvshmem_ulonglong_atomic_xor(&ulonglong_var, ulonglong_var, int_var);
  nvshmem_int32_atomic_xor(&int32_var, int32_var, int_var);
  nvshmem_int64_atomic_xor(&int64_var, int64_var, int_var);
  nvshmem_uint32_atomic_xor(&uint32_var, uint32_var, int_var);
  nvshmem_uint64_atomic_xor(&uint64_var, uint64_var, int_var);

  nvshmem_float_get(&float_var, &const_float_var, size_var, int_var);
  nvshmem_double_get(&double_var, &const_double_var, size_var, int_var);
  nvshmem_char_get(&char_var, &const_char_var, size_var, int_var);
  nvshmem_schar_get(&schar_var, &const_signed_char_var, size_var, int_var);
  nvshmem_short_get(&short_var, &const_short_var, size_var, int_var);
  nvshmem_int_get(&int_var, &const_int_var, size_var, int_var);
  nvshmem_long_get(&long_var, &const_long_var, size_var, int_var);
  nvshmem_longlong_get(&longlong_var, &const_long_long_var, size_var, int_var);
  nvshmem_uchar_get(&uchar_var, &const_unsigned_char_var, size_var, int_var);
  nvshmem_ushort_get(&ushort_var, &const_unsigned_short_var, size_var, int_var);
  nvshmem_uint_get(&uint_var, &const_unsigned_int_var, size_var, int_var);
  nvshmem_ulong_get(&ulong_var, &const_unsigned_long_var, size_var, int_var);
  nvshmem_ulonglong_get(&ulonglong_var, &const_unsigned_long_long_var, size_var, int_var);
  nvshmem_int8_get(&int8_var, &const_int8_t_var, size_var, int_var);
  nvshmem_int16_get(&int16_var, &const_int16_t_var, size_var, int_var);
  nvshmem_int32_get(&int32_var, &const_int32_t_var, size_var, int_var);
  nvshmem_int64_get(&int64_var, &const_int64_t_var, size_var, int_var);
  nvshmem_uint8_get(&uint8_var, &const_uint8_t_var, size_var, int_var);
  nvshmem_uint16_get(&uint16_var, &const_uint16_t_var, size_var, int_var);
  nvshmem_uint32_get(&uint32_var, &const_uint32_t_var, size_var, int_var);
  nvshmem_uint64_get(&uint64_var, &const_uint64_t_var, size_var, int_var);
  nvshmem_size_get(&size_var, &const_size_t_var, size_var, int_var);
  nvshmem_ptrdiff_get(&ptrdiff_var, &const_ptrdiff_t_var, size_var, int_var);

  nvshmem_float_get_nbi(&float_var, &const_float_var, size_var, int_var);
  nvshmem_double_get_nbi(&double_var, &const_double_var, size_var, int_var);
  nvshmem_char_get_nbi(&char_var, &const_char_var, size_var, int_var);
  nvshmem_schar_get_nbi(&schar_var, &const_signed_char_var, size_var, int_var);
  nvshmem_short_get_nbi(&short_var, &const_short_var, size_var, int_var);
  nvshmem_int_get_nbi(&int_var, &const_int_var, size_var, int_var);
  nvshmem_long_get_nbi(&long_var, &const_long_var, size_var, int_var);
  nvshmem_longlong_get_nbi(&longlong_var, &const_long_long_var, size_var, int_var);
  nvshmem_uchar_get_nbi(&uchar_var, &const_unsigned_char_var, size_var, int_var);
  nvshmem_ushort_get_nbi(&ushort_var, &const_unsigned_short_var, size_var, int_var);
  nvshmem_uint_get_nbi(&uint_var, &const_unsigned_int_var, size_var, int_var);
  nvshmem_ulong_get_nbi(&ulong_var, &const_unsigned_long_var, size_var, int_var);
  nvshmem_ulonglong_get_nbi(&ulonglong_var, &const_unsigned_long_long_var, size_var, int_var);
  nvshmem_int8_get_nbi(&int8_var, &const_int8_t_var, size_var, int_var);
  nvshmem_int16_get_nbi(&int16_var, &const_int16_t_var, size_var, int_var);
  nvshmem_int32_get_nbi(&int32_var, &const_int32_t_var, size_var, int_var);
  nvshmem_int64_get_nbi(&int64_var, &const_int64_t_var, size_var, int_var);
  nvshmem_uint8_get_nbi(&uint8_var, &const_uint8_t_var, size_var, int_var);
  nvshmem_uint16_get_nbi(&uint16_var, &const_uint16_t_var, size_var, int_var);
  nvshmem_uint32_get_nbi(&uint32_var, &const_uint32_t_var, size_var, int_var);
  nvshmem_uint64_get_nbi(&uint64_var, &const_uint64_t_var, size_var, int_var);
  nvshmem_size_get_nbi(&size_var, &const_size_t_var, size_var, int_var);
  nvshmem_ptrdiff_get_nbi(&ptrdiff_var, &const_ptrdiff_t_var, size_var, int_var);

  nvshmem_float_iget(&float_var, &const_float_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_double_iget(&double_var, &const_double_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_char_iget(&char_var, &const_char_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_schar_iget(&schar_var, &const_signed_char_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_short_iget(&short_var, &const_short_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_int_iget(&int_var, &const_int_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_long_iget(&long_var, &const_long_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_longlong_iget(&longlong_var, &const_long_long_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_uchar_iget(&uchar_var, &const_unsigned_char_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_ushort_iget(&ushort_var, &const_unsigned_short_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_uint_iget(&uint_var, &const_unsigned_int_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_ulong_iget(&ulong_var, &const_unsigned_long_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_ulonglong_iget(&ulonglong_var, &const_unsigned_long_long_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_int8_iget(&int8_var, &const_int8_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_int16_iget(&int16_var, &const_int16_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_int32_iget(&int32_var, &const_int32_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_int64_iget(&int64_var, &const_int64_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_uint8_iget(&uint8_var, &const_uint8_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_uint16_iget(&uint16_var, &const_uint16_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_uint32_iget(&uint32_var, &const_uint32_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_uint64_iget(&uint64_var, &const_uint64_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_size_iget(&size_var, &const_size_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_ptrdiff_iget(&ptrdiff_var, &const_ptrdiff_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);

  nvshmem_float_iput(&float_var, &const_float_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_double_iput(&double_var, &const_double_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_char_iput(&char_var, &const_char_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_schar_iput(&schar_var, &const_signed_char_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_short_iput(&short_var, &const_short_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_int_iput(&int_var, &const_int_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_long_iput(&long_var, &const_long_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_longlong_iput(&longlong_var, &const_long_long_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_uchar_iput(&uchar_var, &const_unsigned_char_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_ushort_iput(&ushort_var, &const_unsigned_short_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_uint_iput(&uint_var, &const_unsigned_int_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_ulong_iput(&ulong_var, &const_unsigned_long_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_ulonglong_iput(&ulonglong_var, &const_unsigned_long_long_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_int8_iput(&int8_var, &const_int8_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_int16_iput(&int16_var, &const_int16_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_int32_iput(&int32_var, &const_int32_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_int64_iput(&int64_var, &const_int64_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_uint8_iput(&uint8_var, &const_uint8_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_uint16_iput(&uint16_var, &const_uint16_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_uint32_iput(&uint32_var, &const_uint32_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_uint64_iput(&uint64_var, &const_uint64_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_size_iput(&size_var, &const_size_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_ptrdiff_iput(&ptrdiff_var, &const_ptrdiff_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);

  nvshmem_float_p(&float_var, float_var, int_var);
  nvshmem_double_p(&double_var, double_var, int_var);
  nvshmem_char_p(&char_var, char_var, int_var);
  nvshmem_schar_p(&schar_var, schar_var, int_var);
  nvshmem_short_p(&short_var, short_var, int_var);
  nvshmem_int_p(&int_var, int_var, int_var);
  nvshmem_long_p(&long_var, long_var, int_var);
  nvshmem_longlong_p(&longlong_var, longlong_var, int_var);
  nvshmem_uchar_p(&uchar_var, uchar_var, int_var);
  nvshmem_ushort_p(&ushort_var, ushort_var, int_var);
  nvshmem_uint_p(&uint_var, uint_var, int_var);
  nvshmem_ulong_p(&ulong_var, ulong_var, int_var);
  nvshmem_ulonglong_p(&ulonglong_var, ulonglong_var, int_var);
  nvshmem_int8_p(&int8_var, int8_var, int_var);
  nvshmem_int16_p(&int16_var, int16_var, int_var);
  nvshmem_int32_p(&int32_var, int32_var, int_var);
  nvshmem_int64_p(&int64_var, int64_var, int_var);
  nvshmem_uint8_p(&uint8_var, uint8_var, int_var);
  nvshmem_uint16_p(&uint16_var, uint16_var, int_var);
  nvshmem_uint32_p(&uint32_var, uint32_var, int_var);
  nvshmem_uint64_p(&uint64_var, uint64_var, int_var);
  nvshmem_size_p(&size_var, size_var, int_var);
  nvshmem_ptrdiff_p(&ptrdiff_var, ptrdiff_var, int_var);

  /// Remote Memory Access (RMA)
  // nvshmem_TYPENAME_put
  nvshmem_float_put(&float_var, &const_float_var, size_var, int_var);
  nvshmem_double_put(&double_var, &const_double_var, size_var, int_var);
  nvshmem_char_put(&char_var, &const_char_var, size_var, int_var);
  nvshmem_schar_put(&schar_var, &const_signed_char_var, size_var, int_var);
  nvshmem_short_put(&short_var, &const_short_var, size_var, int_var);
  nvshmem_int_put(&int_var, &const_int_var, size_var, int_var);
  nvshmem_long_put(&long_var, &const_long_var, size_var, int_var);
  nvshmem_longlong_put(&longlong_var, &const_long_long_var, size_var, int_var);
  nvshmem_uchar_put(&uchar_var, &const_unsigned_char_var, size_var, int_var);
  nvshmem_ushort_put(&ushort_var, &const_unsigned_short_var, size_var, int_var);
  nvshmem_uint_put(&uint_var, &const_unsigned_int_var, size_var, int_var);
  nvshmem_ulong_put(&ulong_var, &const_unsigned_long_var, size_var, int_var);
  nvshmem_ulonglong_put(&ulonglong_var, &const_unsigned_long_long_var, size_var, int_var);
  nvshmem_int8_put(&int8_var, &const_int8_t_var, size_var, int_var);
  nvshmem_int16_put(&int16_var, &const_int16_t_var, size_var, int_var);
  nvshmem_int32_put(&int32_var, &const_int32_t_var, size_var, int_var);
  nvshmem_int64_put(&int64_var, &const_int64_t_var, size_var, int_var);
  nvshmem_uint8_put(&uint8_var, &const_uint8_t_var, size_var, int_var);
  nvshmem_uint16_put(&uint16_var, &const_uint16_t_var, size_var, int_var);
  nvshmem_uint32_put(&uint32_var, &const_uint32_t_var, size_var, int_var);
  nvshmem_uint64_put(&uint64_var, &const_uint64_t_var, size_var, int_var);
  nvshmem_size_put(&size_var, &const_size_t_var, size_var, int_var);
  nvshmem_ptrdiff_put(&ptrdiff_var, &const_ptrdiff_t_var, size_var, int_var);

  nvshmem_float_put_nbi(&float_var, &const_float_var, size_var, int_var);
  nvshmem_double_put_nbi(&double_var, &const_double_var, size_var, int_var);
  nvshmem_char_put_nbi(&char_var, &const_char_var, size_var, int_var);
  nvshmem_schar_put_nbi(&schar_var, &const_signed_char_var, size_var, int_var);
  nvshmem_short_put_nbi(&short_var, &const_short_var, size_var, int_var);
  nvshmem_int_put_nbi(&int_var, &const_int_var, size_var, int_var);
  nvshmem_long_put_nbi(&long_var, &const_long_var, size_var, int_var);
  nvshmem_longlong_put_nbi(&longlong_var, &const_long_long_var, size_var, int_var);
  nvshmem_uchar_put_nbi(&uchar_var, &const_unsigned_char_var, size_var, int_var);
  nvshmem_ushort_put_nbi(&ushort_var, &const_unsigned_short_var, size_var, int_var);
  nvshmem_uint_put_nbi(&uint_var, &const_unsigned_int_var, size_var, int_var);
  nvshmem_ulong_put_nbi(&ulong_var, &const_unsigned_long_var, size_var, int_var);
  nvshmem_ulonglong_put_nbi(&ulonglong_var, &const_unsigned_long_long_var, size_var, int_var);
  nvshmem_int8_put_nbi(&int8_var, &const_int8_t_var, size_var, int_var);
  nvshmem_int16_put_nbi(&int16_var, &const_int16_t_var, size_var, int_var);
  nvshmem_int32_put_nbi(&int32_var, &const_int32_t_var, size_var, int_var);
  nvshmem_int64_put_nbi(&int64_var, &const_int64_t_var, size_var, int_var);
  nvshmem_uint8_put_nbi(&uint8_var, &const_uint8_t_var, size_var, int_var);
  nvshmem_uint16_put_nbi(&uint16_var, &const_uint16_t_var, size_var, int_var);
  nvshmem_uint32_put_nbi(&uint32_var, &const_uint32_t_var, size_var, int_var);
  nvshmem_uint64_put_nbi(&uint64_var, &const_uint64_t_var, size_var, int_var);
  nvshmem_size_put_nbi(&size_var, &const_size_t_var, size_var, int_var);
  nvshmem_ptrdiff_put_nbi(&ptrdiff_var, &const_ptrdiff_t_var, size_var, int_var);

  nvshmem_float_put_signal(&float_var, &const_float_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_double_put_signal(&double_var, &const_double_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_char_put_signal(&char_var, &const_char_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_schar_put_signal(&schar_var, &const_signed_char_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_short_put_signal(&short_var, &const_short_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_int_put_signal(&int_var, &const_int_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_long_put_signal(&long_var, &const_long_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_longlong_put_signal(&longlong_var, &const_long_long_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_uchar_put_signal(&uchar_var, &const_unsigned_char_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_ushort_put_signal(&ushort_var, &const_unsigned_short_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_uint_put_signal(&uint_var, &const_unsigned_int_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_ulong_put_signal(&ulong_var, &const_unsigned_long_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_ulonglong_put_signal(&ulonglong_var, &const_unsigned_long_long_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_int8_put_signal(&int8_var, &const_int8_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_int16_put_signal(&int16_var, &const_int16_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_int32_put_signal(&int32_var, &const_int32_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_int64_put_signal(&int64_var, &const_int64_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_uint8_put_signal(&uint8_var, &const_uint8_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_uint16_put_signal(&uint16_var, &const_uint16_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_uint32_put_signal(&uint32_var, &const_uint32_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_uint64_put_signal(&uint64_var, &const_uint64_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_size_put_signal(&size_var, &const_size_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_ptrdiff_put_signal(&ptrdiff_var, &const_ptrdiff_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);

  nvshmem_float_put_signal_nbi(&float_var, &const_float_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_double_put_signal_nbi(&double_var, &const_double_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_char_put_signal_nbi(&char_var, &const_char_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_schar_put_signal_nbi(&schar_var, &const_signed_char_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_short_put_signal_nbi(&short_var, &const_short_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_int_put_signal_nbi(&int_var, &const_int_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_long_put_signal_nbi(&long_var, &const_long_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_longlong_put_signal_nbi(&longlong_var, &const_long_long_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_uchar_put_signal_nbi(&uchar_var, &const_unsigned_char_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_ushort_put_signal_nbi(&ushort_var, &const_unsigned_short_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_uint_put_signal_nbi(&uint_var, &const_unsigned_int_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_ulong_put_signal_nbi(&ulong_var, &const_unsigned_long_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_ulonglong_put_signal_nbi(&ulonglong_var, &const_unsigned_long_long_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_int8_put_signal_nbi(&int8_var, &const_int8_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_int16_put_signal_nbi(&int16_var, &const_int16_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_int32_put_signal_nbi(&int32_var, &const_int32_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_int64_put_signal_nbi(&int64_var, &const_int64_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_uint8_put_signal_nbi(&uint8_var, &const_uint8_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_uint16_put_signal_nbi(&uint16_var, &const_uint16_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_uint32_put_signal_nbi(&uint32_var, &const_uint32_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_uint64_put_signal_nbi(&uint64_var, &const_uint64_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_size_put_signal_nbi(&size_var, &const_size_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_ptrdiff_put_signal_nbi(&ptrdiff_var, &const_ptrdiff_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);

  // nvshmem_short_wait(&short_var, short_var);
  // nvshmem_int_wait(&int_var, int_var);
  // nvshmem_long_wait(&long_var, long_var);
  // nvshmem_longlong_wait(&longlong_var, longlong_var);
  // nvshmem_ushort_wait(&ushort_var, ushort_var);
  // nvshmem_uint_wait(&uint_var, uint_var);
  // nvshmem_ulong_wait(&ulong_var, ulong_var);
  // nvshmem_ulonglong_wait(&ulonglong_var, ulonglong_var);
  // nvshmem_int32_wait(&int32_var, int32_var);
  // nvshmem_int64_wait(&int64_var, int64_var);
  // nvshmem_uint32_wait(&uint32_var, uint32_var);
  // nvshmem_uint64_wait(&uint64_var, uint64_var);
  // nvshmem_size_wait(&size_var, size_var);
  // nvshmem_ptrdiff_wait(&ptrdiff_var, ptrdiff_var);

  nvshmem_short_wait_until(&short_var, int_var, short_var);
  nvshmem_int_wait_until(&int_var, int_var, int_var);
  nvshmem_long_wait_until(&long_var, int_var, long_var);
  nvshmem_longlong_wait_until(&longlong_var, int_var, longlong_var);
  nvshmem_ushort_wait_until(&ushort_var, int_var, ushort_var);
  nvshmem_uint_wait_until(&uint_var, int_var, uint_var);
  nvshmem_ulong_wait_until(&ulong_var, int_var, ulong_var);
  nvshmem_ulonglong_wait_until(&ulonglong_var, int_var, ulonglong_var);
  nvshmem_int32_wait_until(&int32_var, int_var, int32_var);
  nvshmem_int64_wait_until(&int64_var, int_var, int64_var);
  nvshmem_uint32_wait_until(&uint32_var, int_var, uint32_var);
  nvshmem_uint64_wait_until(&uint64_var, int_var, uint64_var);
  nvshmem_size_wait_until(&size_var, int_var, size_var);
  nvshmem_ptrdiff_wait_until(&ptrdiff_var, int_var, ptrdiff_var);

  nvshmem_short_wait_until_all(&short_var, size_var, &const_int_var, int_var, short_var);
  nvshmem_int_wait_until_all(&int_var, size_var, &const_int_var, int_var, int_var);
  nvshmem_long_wait_until_all(&long_var, size_var, &const_int_var, int_var, long_var);
  nvshmem_longlong_wait_until_all(&longlong_var, size_var, &const_int_var, int_var, longlong_var);
  nvshmem_ushort_wait_until_all(&ushort_var, size_var, &const_int_var, int_var, ushort_var);
  nvshmem_uint_wait_until_all(&uint_var, size_var, &const_int_var, int_var, uint_var);
  nvshmem_ulong_wait_until_all(&ulong_var, size_var, &const_int_var, int_var, ulong_var);
  nvshmem_ulonglong_wait_until_all(&ulonglong_var, size_var, &const_int_var, int_var, ulonglong_var);
  nvshmem_int32_wait_until_all(&int32_var, size_var, &const_int_var, int_var, int32_var);
  nvshmem_int64_wait_until_all(&int64_var, size_var, &const_int_var, int_var, int64_var);
  nvshmem_uint32_wait_until_all(&uint32_var, size_var, &const_int_var, int_var, uint32_var);
  nvshmem_uint64_wait_until_all(&uint64_var, size_var, &const_int_var, int_var, uint64_var);
  nvshmem_size_wait_until_all(&size_var, size_var, &const_int_var, int_var, size_var);
  nvshmem_ptrdiff_wait_until_all(&ptrdiff_var, size_var, &const_int_var, int_var, ptrdiff_var);

  nvshmem_short_wait_until_all_vector(&short_var, size_var, &const_int_var, int_var, &short_var);
  nvshmem_int_wait_until_all_vector(&int_var, size_var, &const_int_var, int_var, &int_var);
  nvshmem_long_wait_until_all_vector(&long_var, size_var, &const_int_var, int_var, &long_var);
  nvshmem_longlong_wait_until_all_vector(&longlong_var, size_var, &const_int_var, int_var, &longlong_var);
  nvshmem_ushort_wait_until_all_vector(&ushort_var, size_var, &const_int_var, int_var, &ushort_var);
  nvshmem_uint_wait_until_all_vector(&uint_var, size_var, &const_int_var, int_var, &uint_var);
  nvshmem_ulong_wait_until_all_vector(&ulong_var, size_var, &const_int_var, int_var, &ulong_var);
  nvshmem_ulonglong_wait_until_all_vector(&ulonglong_var, size_var, &const_int_var, int_var, &ulonglong_var);
  nvshmem_int32_wait_until_all_vector(&int32_var, size_var, &const_int_var, int_var, &int32_var);
  nvshmem_int64_wait_until_all_vector(&int64_var, size_var, &const_int_var, int_var, &int64_var);
  nvshmem_uint32_wait_until_all_vector(&uint32_var, size_var, &const_int_var, int_var, &uint32_var);
  nvshmem_uint64_wait_until_all_vector(&uint64_var, size_var, &const_int_var, int_var, &uint64_var);
  nvshmem_size_wait_until_all_vector(&size_var, size_var, &const_int_var, int_var, &size_var);
  nvshmem_ptrdiff_wait_until_all_vector(&ptrdiff_var, size_var, &const_int_var, int_var, &ptrdiff_var);

  nvshmem_barrier(nvshmem_team_t_var);

  nvshmem_barrier_all();

  nvshmem_fence();

  // nvshmem_getSIZE (8, 16, 32, 64, 128)
  nvshmem_get8(void_ptr_var, const_void_var, size_var, int_var);
  nvshmem_get16(void_ptr_var, const_void_var, size_var, int_var);
  nvshmem_get32(void_ptr_var, const_void_var, size_var, int_var);
  nvshmem_get64(void_ptr_var, const_void_var, size_var, int_var);
  nvshmem_get128(void_ptr_var, const_void_var, size_var, int_var);

  // nvshmem_getSIZE_nbi (8, 16, 32, 64, 128)
  nvshmem_get8_nbi(void_ptr_var, const_void_var, size_var, int_var);
  nvshmem_get16_nbi(void_ptr_var, const_void_var, size_var, int_var);
  nvshmem_get32_nbi(void_ptr_var, const_void_var, size_var, int_var);
  nvshmem_get64_nbi(void_ptr_var, const_void_var, size_var, int_var);
  nvshmem_get128_nbi(void_ptr_var, const_void_var, size_var, int_var);

  nvshmem_getmem(void_ptr_var, const_void_var, size_var, int_var);
  nvshmem_getmem_nbi(void_ptr_var, const_void_var, size_var, int_var);

  // nvshmem_igetSIZE (8, 16, 32, 64, 128)
  nvshmem_iget8(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_iget16(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_iget32(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_iget64(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_iget128(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var);

  // nvshmem_iputSIZE (8, 16, 32, 64, 128)
  nvshmem_iput8(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_iput16(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_iput32(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_iput64(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmem_iput128(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var);

  // nvshmem_putSIZE (8, 16, 32, 64, 128)
  nvshmem_put8(void_ptr_var, const_void_var, size_var, int_var);
  nvshmem_put16(void_ptr_var, const_void_var, size_var, int_var);
  nvshmem_put32(void_ptr_var, const_void_var, size_var, int_var);
  nvshmem_put64(void_ptr_var, const_void_var, size_var, int_var);
  nvshmem_put128(void_ptr_var, const_void_var, size_var, int_var);

  // nvshmem_putSIZE_nbi (8, 16, 32, 64, 128)
  nvshmem_put8_nbi(void_ptr_var, const_void_var, size_var, int_var);
  nvshmem_put16_nbi(void_ptr_var, const_void_var, size_var, int_var);
  nvshmem_put32_nbi(void_ptr_var, const_void_var, size_var, int_var);
  nvshmem_put64_nbi(void_ptr_var, const_void_var, size_var, int_var);
  nvshmem_put128_nbi(void_ptr_var, const_void_var, size_var, int_var);

  // nvshmem_putSIZE_signal (8, 16, 32, 64, 128)
  nvshmem_put8_signal(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_put16_signal(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_put32_signal(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_put64_signal(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_put128_signal(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var);

  // nvshmem_putSIZE_signal_nbi (8, 16, 32, 64, 128)
  nvshmem_put8_signal_nbi(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_put16_signal_nbi(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_put32_signal_nbi(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_put64_signal_nbi(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_put128_signal_nbi(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var);

  nvshmem_putmem(void_ptr_var, const_void_var, size_var, int_var);
  nvshmem_putmem_nbi(void_ptr_var, const_void_var, size_var, int_var);
  nvshmem_putmem_signal(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmem_putmem_signal_nbi(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var);

  nvshmem_quiet();
  nvshmem_sync_all();

  // nvshmemx_TYPENAME_get_block (RMA)
  nvshmemx_float_get_block(&float_var, &const_float_var, size_var, int_var);
  nvshmemx_double_get_block(&double_var, &const_double_var, size_var, int_var);
  nvshmemx_char_get_block(&char_var, &const_char_var, size_var, int_var);
  nvshmemx_schar_get_block(&schar_var, &const_signed_char_var, size_var, int_var);
  nvshmemx_short_get_block(&short_var, &const_short_var, size_var, int_var);
  nvshmemx_int_get_block(&int_var, &const_int_var, size_var, int_var);
  nvshmemx_long_get_block(&long_var, &const_long_var, size_var, int_var);
  nvshmemx_longlong_get_block(&longlong_var, &const_long_long_var, size_var, int_var);
  nvshmemx_uchar_get_block(&uchar_var, &const_unsigned_char_var, size_var, int_var);
  nvshmemx_ushort_get_block(&ushort_var, &const_unsigned_short_var, size_var, int_var);
  nvshmemx_uint_get_block(&uint_var, &const_unsigned_int_var, size_var, int_var);
  nvshmemx_ulong_get_block(&ulong_var, &const_unsigned_long_var, size_var, int_var);
  nvshmemx_ulonglong_get_block(&ulonglong_var, &const_unsigned_long_long_var, size_var, int_var);
  nvshmemx_int8_get_block(&int8_var, &const_int8_t_var, size_var, int_var);
  nvshmemx_int16_get_block(&int16_var, &const_int16_t_var, size_var, int_var);
  nvshmemx_int32_get_block(&int32_var, &const_int32_t_var, size_var, int_var);
  nvshmemx_int64_get_block(&int64_var, &const_int64_t_var, size_var, int_var);
  nvshmemx_uint8_get_block(&uint8_var, &const_uint8_t_var, size_var, int_var);
  nvshmemx_uint16_get_block(&uint16_var, &const_uint16_t_var, size_var, int_var);
  nvshmemx_uint32_get_block(&uint32_var, &const_uint32_t_var, size_var, int_var);
  nvshmemx_uint64_get_block(&uint64_var, &const_uint64_t_var, size_var, int_var);
  nvshmemx_size_get_block(&size_var, &const_size_t_var, size_var, int_var);
  nvshmemx_ptrdiff_get_block(&ptrdiff_var, &const_ptrdiff_t_var, size_var, int_var);

  // nvshmemx_TYPENAME_get_nbi_block (RMA)
  nvshmemx_float_get_nbi_block(&float_var, &const_float_var, size_var, int_var);
  nvshmemx_double_get_nbi_block(&double_var, &const_double_var, size_var, int_var);
  nvshmemx_char_get_nbi_block(&char_var, &const_char_var, size_var, int_var);
  nvshmemx_schar_get_nbi_block(&schar_var, &const_signed_char_var, size_var, int_var);
  nvshmemx_short_get_nbi_block(&short_var, &const_short_var, size_var, int_var);
  nvshmemx_int_get_nbi_block(&int_var, &const_int_var, size_var, int_var);
  nvshmemx_long_get_nbi_block(&long_var, &const_long_var, size_var, int_var);
  nvshmemx_longlong_get_nbi_block(&longlong_var, &const_long_long_var, size_var, int_var);
  nvshmemx_uchar_get_nbi_block(&uchar_var, &const_unsigned_char_var, size_var, int_var);
  nvshmemx_ushort_get_nbi_block(&ushort_var, &const_unsigned_short_var, size_var, int_var);
  nvshmemx_uint_get_nbi_block(&uint_var, &const_unsigned_int_var, size_var, int_var);
  nvshmemx_ulong_get_nbi_block(&ulong_var, &const_unsigned_long_var, size_var, int_var);
  nvshmemx_ulonglong_get_nbi_block(&ulonglong_var, &const_unsigned_long_long_var, size_var, int_var);
  nvshmemx_int8_get_nbi_block(&int8_var, &const_int8_t_var, size_var, int_var);
  nvshmemx_int16_get_nbi_block(&int16_var, &const_int16_t_var, size_var, int_var);
  nvshmemx_int32_get_nbi_block(&int32_var, &const_int32_t_var, size_var, int_var);
  nvshmemx_int64_get_nbi_block(&int64_var, &const_int64_t_var, size_var, int_var);
  nvshmemx_uint8_get_nbi_block(&uint8_var, &const_uint8_t_var, size_var, int_var);
  nvshmemx_uint16_get_nbi_block(&uint16_var, &const_uint16_t_var, size_var, int_var);
  nvshmemx_uint32_get_nbi_block(&uint32_var, &const_uint32_t_var, size_var, int_var);
  nvshmemx_uint64_get_nbi_block(&uint64_var, &const_uint64_t_var, size_var, int_var);
  nvshmemx_size_get_nbi_block(&size_var, &const_size_t_var, size_var, int_var);
  nvshmemx_ptrdiff_get_nbi_block(&ptrdiff_var, &const_ptrdiff_t_var, size_var, int_var);

  // nvshmemx_TYPENAME_get_nbi_warp (RMA)
  nvshmemx_float_get_nbi_warp(&float_var, &const_float_var, size_var, int_var);
  nvshmemx_double_get_nbi_warp(&double_var, &const_double_var, size_var, int_var);
  nvshmemx_char_get_nbi_warp(&char_var, &const_char_var, size_var, int_var);
  nvshmemx_schar_get_nbi_warp(&schar_var, &const_signed_char_var, size_var, int_var);
  nvshmemx_short_get_nbi_warp(&short_var, &const_short_var, size_var, int_var);
  nvshmemx_int_get_nbi_warp(&int_var, &const_int_var, size_var, int_var);
  nvshmemx_long_get_nbi_warp(&long_var, &const_long_var, size_var, int_var);
  nvshmemx_longlong_get_nbi_warp(&longlong_var, &const_long_long_var, size_var, int_var);
  nvshmemx_uchar_get_nbi_warp(&uchar_var, &const_unsigned_char_var, size_var, int_var);
  nvshmemx_ushort_get_nbi_warp(&ushort_var, &const_unsigned_short_var, size_var, int_var);
  nvshmemx_uint_get_nbi_warp(&uint_var, &const_unsigned_int_var, size_var, int_var);
  nvshmemx_ulong_get_nbi_warp(&ulong_var, &const_unsigned_long_var, size_var, int_var);
  nvshmemx_ulonglong_get_nbi_warp(&ulonglong_var, &const_unsigned_long_long_var, size_var, int_var);
  nvshmemx_int8_get_nbi_warp(&int8_var, &const_int8_t_var, size_var, int_var);
  nvshmemx_int16_get_nbi_warp(&int16_var, &const_int16_t_var, size_var, int_var);
  nvshmemx_int32_get_nbi_warp(&int32_var, &const_int32_t_var, size_var, int_var);
  nvshmemx_int64_get_nbi_warp(&int64_var, &const_int64_t_var, size_var, int_var);
  nvshmemx_uint8_get_nbi_warp(&uint8_var, &const_uint8_t_var, size_var, int_var);
  nvshmemx_uint16_get_nbi_warp(&uint16_var, &const_uint16_t_var, size_var, int_var);
  nvshmemx_uint32_get_nbi_warp(&uint32_var, &const_uint32_t_var, size_var, int_var);
  nvshmemx_uint64_get_nbi_warp(&uint64_var, &const_uint64_t_var, size_var, int_var);
  nvshmemx_size_get_nbi_warp(&size_var, &const_size_t_var, size_var, int_var);
  nvshmemx_ptrdiff_get_nbi_warp(&ptrdiff_var, &const_ptrdiff_t_var, size_var, int_var);

  // nvshmemx_TYPENAME_get_warp (RMA)
  nvshmemx_float_get_warp(&float_var, &const_float_var, size_var, int_var);
  nvshmemx_double_get_warp(&double_var, &const_double_var, size_var, int_var);
  nvshmemx_char_get_warp(&char_var, &const_char_var, size_var, int_var);
  nvshmemx_schar_get_warp(&schar_var, &const_signed_char_var, size_var, int_var);
  nvshmemx_short_get_warp(&short_var, &const_short_var, size_var, int_var);
  nvshmemx_int_get_warp(&int_var, &const_int_var, size_var, int_var);
  nvshmemx_long_get_warp(&long_var, &const_long_var, size_var, int_var);
  nvshmemx_longlong_get_warp(&longlong_var, &const_long_long_var, size_var, int_var);
  nvshmemx_uchar_get_warp(&uchar_var, &const_unsigned_char_var, size_var, int_var);
  nvshmemx_ushort_get_warp(&ushort_var, &const_unsigned_short_var, size_var, int_var);
  nvshmemx_uint_get_warp(&uint_var, &const_unsigned_int_var, size_var, int_var);
  nvshmemx_ulong_get_warp(&ulong_var, &const_unsigned_long_var, size_var, int_var);
  nvshmemx_ulonglong_get_warp(&ulonglong_var, &const_unsigned_long_long_var, size_var, int_var);
  nvshmemx_int8_get_warp(&int8_var, &const_int8_t_var, size_var, int_var);
  nvshmemx_int16_get_warp(&int16_var, &const_int16_t_var, size_var, int_var);
  nvshmemx_int32_get_warp(&int32_var, &const_int32_t_var, size_var, int_var);
  nvshmemx_int64_get_warp(&int64_var, &const_int64_t_var, size_var, int_var);
  nvshmemx_uint8_get_warp(&uint8_var, &const_uint8_t_var, size_var, int_var);
  nvshmemx_uint16_get_warp(&uint16_var, &const_uint16_t_var, size_var, int_var);
  nvshmemx_uint32_get_warp(&uint32_var, &const_uint32_t_var, size_var, int_var);
  nvshmemx_uint64_get_warp(&uint64_var, &const_uint64_t_var, size_var, int_var);
  nvshmemx_size_get_warp(&size_var, &const_size_t_var, size_var, int_var);
  nvshmemx_ptrdiff_get_warp(&ptrdiff_var, &const_ptrdiff_t_var, size_var, int_var);

  // nvshmemx_TYPENAME_iget_block (RMA)
  nvshmemx_float_iget_block(&float_var, &const_float_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_double_iget_block(&double_var, &const_double_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_char_iget_block(&char_var, &const_char_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_schar_iget_block(&schar_var, &const_signed_char_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_short_iget_block(&short_var, &const_short_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_int_iget_block(&int_var, &const_int_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_long_iget_block(&long_var, &const_long_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_longlong_iget_block(&longlong_var, &const_long_long_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_uchar_iget_block(&uchar_var, &const_unsigned_char_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_ushort_iget_block(&ushort_var, &const_unsigned_short_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_uint_iget_block(&uint_var, &const_unsigned_int_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_ulong_iget_block(&ulong_var, &const_unsigned_long_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_ulonglong_iget_block(&ulonglong_var, &const_unsigned_long_long_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_int8_iget_block(&int8_var, &const_int8_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_int16_iget_block(&int16_var, &const_int16_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_int32_iget_block(&int32_var, &const_int32_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_int64_iget_block(&int64_var, &const_int64_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_uint8_iget_block(&uint8_var, &const_uint8_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_uint16_iget_block(&uint16_var, &const_uint16_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_uint32_iget_block(&uint32_var, &const_uint32_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_uint64_iget_block(&uint64_var, &const_uint64_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_size_iget_block(&size_var, &const_size_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_ptrdiff_iget_block(&ptrdiff_var, &const_ptrdiff_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);

  // nvshmemx_TYPENAME_iget_warp (RMA)
  nvshmemx_float_iget_warp(&float_var, &const_float_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_double_iget_warp(&double_var, &const_double_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_char_iget_warp(&char_var, &const_char_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_schar_iget_warp(&schar_var, &const_signed_char_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_short_iget_warp(&short_var, &const_short_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_int_iget_warp(&int_var, &const_int_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_long_iget_warp(&long_var, &const_long_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_longlong_iget_warp(&longlong_var, &const_long_long_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_uchar_iget_warp(&uchar_var, &const_unsigned_char_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_ushort_iget_warp(&ushort_var, &const_unsigned_short_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_uint_iget_warp(&uint_var, &const_unsigned_int_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_ulong_iget_warp(&ulong_var, &const_unsigned_long_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_ulonglong_iget_warp(&ulonglong_var, &const_unsigned_long_long_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_int8_iget_warp(&int8_var, &const_int8_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_int16_iget_warp(&int16_var, &const_int16_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_int32_iget_warp(&int32_var, &const_int32_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_int64_iget_warp(&int64_var, &const_int64_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_uint8_iget_warp(&uint8_var, &const_uint8_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_uint16_iget_warp(&uint16_var, &const_uint16_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_uint32_iget_warp(&uint32_var, &const_uint32_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_uint64_iget_warp(&uint64_var, &const_uint64_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_size_iget_warp(&size_var, &const_size_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_ptrdiff_iget_warp(&ptrdiff_var, &const_ptrdiff_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);

  // nvshmemx_TYPENAME_iput_block (RMA)
  nvshmemx_float_iput_block(&float_var, &const_float_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_double_iput_block(&double_var, &const_double_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_char_iput_block(&char_var, &const_char_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_schar_iput_block(&schar_var, &const_signed_char_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_short_iput_block(&short_var, &const_short_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_int_iput_block(&int_var, &const_int_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_long_iput_block(&long_var, &const_long_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_longlong_iput_block(&longlong_var, &const_long_long_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_uchar_iput_block(&uchar_var, &const_unsigned_char_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_ushort_iput_block(&ushort_var, &const_unsigned_short_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_uint_iput_block(&uint_var, &const_unsigned_int_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_ulong_iput_block(&ulong_var, &const_unsigned_long_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_ulonglong_iput_block(&ulonglong_var, &const_unsigned_long_long_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_int8_iput_block(&int8_var, &const_int8_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_int16_iput_block(&int16_var, &const_int16_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_int32_iput_block(&int32_var, &const_int32_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_int64_iput_block(&int64_var, &const_int64_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_uint8_iput_block(&uint8_var, &const_uint8_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_uint16_iput_block(&uint16_var, &const_uint16_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_uint32_iput_block(&uint32_var, &const_uint32_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_uint64_iput_block(&uint64_var, &const_uint64_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_size_iput_block(&size_var, &const_size_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_ptrdiff_iput_block(&ptrdiff_var, &const_ptrdiff_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);

  // nvshmemx_TYPENAME_iput_warp (RMA)
  nvshmemx_float_iput_warp(&float_var, &const_float_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_double_iput_warp(&double_var, &const_double_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_char_iput_warp(&char_var, &const_char_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_schar_iput_warp(&schar_var, &const_signed_char_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_short_iput_warp(&short_var, &const_short_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_int_iput_warp(&int_var, &const_int_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_long_iput_warp(&long_var, &const_long_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_longlong_iput_warp(&longlong_var, &const_long_long_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_uchar_iput_warp(&uchar_var, &const_unsigned_char_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_ushort_iput_warp(&ushort_var, &const_unsigned_short_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_uint_iput_warp(&uint_var, &const_unsigned_int_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_ulong_iput_warp(&ulong_var, &const_unsigned_long_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_ulonglong_iput_warp(&ulonglong_var, &const_unsigned_long_long_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_int8_iput_warp(&int8_var, &const_int8_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_int16_iput_warp(&int16_var, &const_int16_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_int32_iput_warp(&int32_var, &const_int32_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_int64_iput_warp(&int64_var, &const_int64_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_uint8_iput_warp(&uint8_var, &const_uint8_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_uint16_iput_warp(&uint16_var, &const_uint16_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_uint32_iput_warp(&uint32_var, &const_uint32_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_uint64_iput_warp(&uint64_var, &const_uint64_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_size_iput_warp(&size_var, &const_size_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_ptrdiff_iput_warp(&ptrdiff_var, &const_ptrdiff_t_var, ptrdiff_var, ptrdiff_var, size_var, int_var);

  // nvshmemx_TYPENAME_put_block (RMA)
  nvshmemx_float_put_block(&float_var, &const_float_var, size_var, int_var);
  nvshmemx_double_put_block(&double_var, &const_double_var, size_var, int_var);
  nvshmemx_char_put_block(&char_var, &const_char_var, size_var, int_var);
  nvshmemx_schar_put_block(&schar_var, &const_signed_char_var, size_var, int_var);
  nvshmemx_short_put_block(&short_var, &const_short_var, size_var, int_var);
  nvshmemx_int_put_block(&int_var, &const_int_var, size_var, int_var);
  nvshmemx_long_put_block(&long_var, &const_long_var, size_var, int_var);
  nvshmemx_longlong_put_block(&longlong_var, &const_long_long_var, size_var, int_var);
  nvshmemx_uchar_put_block(&uchar_var, &const_unsigned_char_var, size_var, int_var);
  nvshmemx_ushort_put_block(&ushort_var, &const_unsigned_short_var, size_var, int_var);
  nvshmemx_uint_put_block(&uint_var, &const_unsigned_int_var, size_var, int_var);
  nvshmemx_ulong_put_block(&ulong_var, &const_unsigned_long_var, size_var, int_var);
  nvshmemx_ulonglong_put_block(&ulonglong_var, &const_unsigned_long_long_var, size_var, int_var);
  nvshmemx_int8_put_block(&int8_var, &const_int8_t_var, size_var, int_var);
  nvshmemx_int16_put_block(&int16_var, &const_int16_t_var, size_var, int_var);
  nvshmemx_int32_put_block(&int32_var, &const_int32_t_var, size_var, int_var);
  nvshmemx_int64_put_block(&int64_var, &const_int64_t_var, size_var, int_var);
  nvshmemx_uint8_put_block(&uint8_var, &const_uint8_t_var, size_var, int_var);
  nvshmemx_uint16_put_block(&uint16_var, &const_uint16_t_var, size_var, int_var);
  nvshmemx_uint32_put_block(&uint32_var, &const_uint32_t_var, size_var, int_var);
  nvshmemx_uint64_put_block(&uint64_var, &const_uint64_t_var, size_var, int_var);
  nvshmemx_size_put_block(&size_var, &const_size_t_var, size_var, int_var);
  nvshmemx_ptrdiff_put_block(&ptrdiff_var, &const_ptrdiff_t_var, size_var, int_var);

  // nvshmemx_TYPENAME_put_nbi_block (RMA)
  nvshmemx_float_put_nbi_block(&float_var, &const_float_var, size_var, int_var);
  nvshmemx_double_put_nbi_block(&double_var, &const_double_var, size_var, int_var);
  nvshmemx_char_put_nbi_block(&char_var, &const_char_var, size_var, int_var);
  nvshmemx_schar_put_nbi_block(&schar_var, &const_signed_char_var, size_var, int_var);
  nvshmemx_short_put_nbi_block(&short_var, &const_short_var, size_var, int_var);
  nvshmemx_int_put_nbi_block(&int_var, &const_int_var, size_var, int_var);
  nvshmemx_long_put_nbi_block(&long_var, &const_long_var, size_var, int_var);
  nvshmemx_longlong_put_nbi_block(&longlong_var, &const_long_long_var, size_var, int_var);
  nvshmemx_uchar_put_nbi_block(&uchar_var, &const_unsigned_char_var, size_var, int_var);
  nvshmemx_ushort_put_nbi_block(&ushort_var, &const_unsigned_short_var, size_var, int_var);
  nvshmemx_uint_put_nbi_block(&uint_var, &const_unsigned_int_var, size_var, int_var);
  nvshmemx_ulong_put_nbi_block(&ulong_var, &const_unsigned_long_var, size_var, int_var);
  nvshmemx_ulonglong_put_nbi_block(&ulonglong_var, &const_unsigned_long_long_var, size_var, int_var);
  nvshmemx_int8_put_nbi_block(&int8_var, &const_int8_t_var, size_var, int_var);
  nvshmemx_int16_put_nbi_block(&int16_var, &const_int16_t_var, size_var, int_var);
  nvshmemx_int32_put_nbi_block(&int32_var, &const_int32_t_var, size_var, int_var);
  nvshmemx_int64_put_nbi_block(&int64_var, &const_int64_t_var, size_var, int_var);
  nvshmemx_uint8_put_nbi_block(&uint8_var, &const_uint8_t_var, size_var, int_var);
  nvshmemx_uint16_put_nbi_block(&uint16_var, &const_uint16_t_var, size_var, int_var);
  nvshmemx_uint32_put_nbi_block(&uint32_var, &const_uint32_t_var, size_var, int_var);
  nvshmemx_uint64_put_nbi_block(&uint64_var, &const_uint64_t_var, size_var, int_var);
  nvshmemx_size_put_nbi_block(&size_var, &const_size_t_var, size_var, int_var);
  nvshmemx_ptrdiff_put_nbi_block(&ptrdiff_var, &const_ptrdiff_t_var, size_var, int_var);

  // nvshmemx_TYPENAME_put_nbi_warp (RMA)
  nvshmemx_float_put_nbi_warp(&float_var, &const_float_var, size_var, int_var);
  nvshmemx_double_put_nbi_warp(&double_var, &const_double_var, size_var, int_var);
  nvshmemx_char_put_nbi_warp(&char_var, &const_char_var, size_var, int_var);
  nvshmemx_schar_put_nbi_warp(&schar_var, &const_signed_char_var, size_var, int_var);
  nvshmemx_short_put_nbi_warp(&short_var, &const_short_var, size_var, int_var);
  nvshmemx_int_put_nbi_warp(&int_var, &const_int_var, size_var, int_var);
  nvshmemx_long_put_nbi_warp(&long_var, &const_long_var, size_var, int_var);
  nvshmemx_longlong_put_nbi_warp(&longlong_var, &const_long_long_var, size_var, int_var);
  nvshmemx_uchar_put_nbi_warp(&uchar_var, &const_unsigned_char_var, size_var, int_var);
  nvshmemx_ushort_put_nbi_warp(&ushort_var, &const_unsigned_short_var, size_var, int_var);
  nvshmemx_uint_put_nbi_warp(&uint_var, &const_unsigned_int_var, size_var, int_var);
  nvshmemx_ulong_put_nbi_warp(&ulong_var, &const_unsigned_long_var, size_var, int_var);
  nvshmemx_ulonglong_put_nbi_warp(&ulonglong_var, &const_unsigned_long_long_var, size_var, int_var);
  nvshmemx_int8_put_nbi_warp(&int8_var, &const_int8_t_var, size_var, int_var);
  nvshmemx_int16_put_nbi_warp(&int16_var, &const_int16_t_var, size_var, int_var);
  nvshmemx_int32_put_nbi_warp(&int32_var, &const_int32_t_var, size_var, int_var);
  nvshmemx_int64_put_nbi_warp(&int64_var, &const_int64_t_var, size_var, int_var);
  nvshmemx_uint8_put_nbi_warp(&uint8_var, &const_uint8_t_var, size_var, int_var);
  nvshmemx_uint16_put_nbi_warp(&uint16_var, &const_uint16_t_var, size_var, int_var);
  nvshmemx_uint32_put_nbi_warp(&uint32_var, &const_uint32_t_var, size_var, int_var);
  nvshmemx_uint64_put_nbi_warp(&uint64_var, &const_uint64_t_var, size_var, int_var);
  nvshmemx_size_put_nbi_warp(&size_var, &const_size_t_var, size_var, int_var);
  nvshmemx_ptrdiff_put_nbi_warp(&ptrdiff_var, &const_ptrdiff_t_var, size_var, int_var);

  // nvshmemx_TYPENAME_put_signal_block (RMA)
  nvshmemx_float_put_signal_block(&float_var, &const_float_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_double_put_signal_block(&double_var, &const_double_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_char_put_signal_block(&char_var, &const_char_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_schar_put_signal_block(&schar_var, &const_signed_char_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_short_put_signal_block(&short_var, &const_short_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_int_put_signal_block(&int_var, &const_int_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_long_put_signal_block(&long_var, &const_long_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_longlong_put_signal_block(&longlong_var, &const_long_long_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_uchar_put_signal_block(&uchar_var, &const_unsigned_char_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_ushort_put_signal_block(&ushort_var, &const_unsigned_short_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_uint_put_signal_block(&uint_var, &const_unsigned_int_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_ulong_put_signal_block(&ulong_var, &const_unsigned_long_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_ulonglong_put_signal_block(&ulonglong_var, &const_unsigned_long_long_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_int8_put_signal_block(&int8_var, &const_int8_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_int16_put_signal_block(&int16_var, &const_int16_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_int32_put_signal_block(&int32_var, &const_int32_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_int64_put_signal_block(&int64_var, &const_int64_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_uint8_put_signal_block(&uint8_var, &const_uint8_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_uint16_put_signal_block(&uint16_var, &const_uint16_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_uint32_put_signal_block(&uint32_var, &const_uint32_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_uint64_put_signal_block(&uint64_var, &const_uint64_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_size_put_signal_block(&size_var, &const_size_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_ptrdiff_put_signal_block(&ptrdiff_var, &const_ptrdiff_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);

  // nvshmemx_TYPENAME_put_signal_nbi_block (RMA)
  nvshmemx_float_put_signal_nbi_block(&float_var, &const_float_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_double_put_signal_nbi_block(&double_var, &const_double_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_char_put_signal_nbi_block(&char_var, &const_char_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_schar_put_signal_nbi_block(&schar_var, &const_signed_char_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_short_put_signal_nbi_block(&short_var, &const_short_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_int_put_signal_nbi_block(&int_var, &const_int_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_long_put_signal_nbi_block(&long_var, &const_long_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_longlong_put_signal_nbi_block(&longlong_var, &const_long_long_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_uchar_put_signal_nbi_block(&uchar_var, &const_unsigned_char_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_ushort_put_signal_nbi_block(&ushort_var, &const_unsigned_short_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_uint_put_signal_nbi_block(&uint_var, &const_unsigned_int_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_ulong_put_signal_nbi_block(&ulong_var, &const_unsigned_long_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_ulonglong_put_signal_nbi_block(&ulonglong_var, &const_unsigned_long_long_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_int8_put_signal_nbi_block(&int8_var, &const_int8_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_int16_put_signal_nbi_block(&int16_var, &const_int16_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_int32_put_signal_nbi_block(&int32_var, &const_int32_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_int64_put_signal_nbi_block(&int64_var, &const_int64_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_uint8_put_signal_nbi_block(&uint8_var, &const_uint8_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_uint16_put_signal_nbi_block(&uint16_var, &const_uint16_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_uint32_put_signal_nbi_block(&uint32_var, &const_uint32_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_uint64_put_signal_nbi_block(&uint64_var, &const_uint64_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_size_put_signal_nbi_block(&size_var, &const_size_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_ptrdiff_put_signal_nbi_block(&ptrdiff_var, &const_ptrdiff_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);

  // nvshmemx_TYPENAME_put_signal_nbi_warp (RMA)
  nvshmemx_float_put_signal_nbi_warp(&float_var, &const_float_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_double_put_signal_nbi_warp(&double_var, &const_double_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_char_put_signal_nbi_warp(&char_var, &const_char_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_schar_put_signal_nbi_warp(&schar_var, &const_signed_char_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_short_put_signal_nbi_warp(&short_var, &const_short_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_int_put_signal_nbi_warp(&int_var, &const_int_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_long_put_signal_nbi_warp(&long_var, &const_long_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_longlong_put_signal_nbi_warp(&longlong_var, &const_long_long_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_uchar_put_signal_nbi_warp(&uchar_var, &const_unsigned_char_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_ushort_put_signal_nbi_warp(&ushort_var, &const_unsigned_short_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_uint_put_signal_nbi_warp(&uint_var, &const_unsigned_int_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_ulong_put_signal_nbi_warp(&ulong_var, &const_unsigned_long_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_ulonglong_put_signal_nbi_warp(&ulonglong_var, &const_unsigned_long_long_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_int8_put_signal_nbi_warp(&int8_var, &const_int8_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_int16_put_signal_nbi_warp(&int16_var, &const_int16_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_int32_put_signal_nbi_warp(&int32_var, &const_int32_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_int64_put_signal_nbi_warp(&int64_var, &const_int64_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_uint8_put_signal_nbi_warp(&uint8_var, &const_uint8_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_uint16_put_signal_nbi_warp(&uint16_var, &const_uint16_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_uint32_put_signal_nbi_warp(&uint32_var, &const_uint32_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_uint64_put_signal_nbi_warp(&uint64_var, &const_uint64_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_size_put_signal_nbi_warp(&size_var, &const_size_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_ptrdiff_put_signal_nbi_warp(&ptrdiff_var, &const_ptrdiff_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);

  // nvshmemx_TYPENAME_put_signal_warp (RMA)
  nvshmemx_float_put_signal_warp(&float_var, &const_float_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_double_put_signal_warp(&double_var, &const_double_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_char_put_signal_warp(&char_var, &const_char_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_schar_put_signal_warp(&schar_var, &const_signed_char_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_short_put_signal_warp(&short_var, &const_short_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_int_put_signal_warp(&int_var, &const_int_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_long_put_signal_warp(&long_var, &const_long_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_longlong_put_signal_warp(&longlong_var, &const_long_long_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_uchar_put_signal_warp(&uchar_var, &const_unsigned_char_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_ushort_put_signal_warp(&ushort_var, &const_unsigned_short_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_uint_put_signal_warp(&uint_var, &const_unsigned_int_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_ulong_put_signal_warp(&ulong_var, &const_unsigned_long_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_ulonglong_put_signal_warp(&ulonglong_var, &const_unsigned_long_long_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_int8_put_signal_warp(&int8_var, &const_int8_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_int16_put_signal_warp(&int16_var, &const_int16_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_int32_put_signal_warp(&int32_var, &const_int32_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_int64_put_signal_warp(&int64_var, &const_int64_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_uint8_put_signal_warp(&uint8_var, &const_uint8_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_uint16_put_signal_warp(&uint16_var, &const_uint16_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_uint32_put_signal_warp(&uint32_var, &const_uint32_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_uint64_put_signal_warp(&uint64_var, &const_uint64_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_size_put_signal_warp(&size_var, &const_size_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_ptrdiff_put_signal_warp(&ptrdiff_var, &const_ptrdiff_t_var, size_var, &uint64_var, uint64_var, int_var, int_var);

  // nvshmemx_TYPENAME_put_warp (RMA)
  nvshmemx_float_put_warp(&float_var, &const_float_var, size_var, int_var);
  nvshmemx_double_put_warp(&double_var, &const_double_var, size_var, int_var);
  nvshmemx_char_put_warp(&char_var, &const_char_var, size_var, int_var);
  nvshmemx_schar_put_warp(&schar_var, &const_signed_char_var, size_var, int_var);
  nvshmemx_short_put_warp(&short_var, &const_short_var, size_var, int_var);
  nvshmemx_int_put_warp(&int_var, &const_int_var, size_var, int_var);
  nvshmemx_long_put_warp(&long_var, &const_long_var, size_var, int_var);
  nvshmemx_longlong_put_warp(&longlong_var, &const_long_long_var, size_var, int_var);
  nvshmemx_uchar_put_warp(&uchar_var, &const_unsigned_char_var, size_var, int_var);
  nvshmemx_ushort_put_warp(&ushort_var, &const_unsigned_short_var, size_var, int_var);
  nvshmemx_uint_put_warp(&uint_var, &const_unsigned_int_var, size_var, int_var);
  nvshmemx_ulong_put_warp(&ulong_var, &const_unsigned_long_var, size_var, int_var);
  nvshmemx_ulonglong_put_warp(&ulonglong_var, &const_unsigned_long_long_var, size_var, int_var);
  nvshmemx_int8_put_warp(&int8_var, &const_int8_t_var, size_var, int_var);
  nvshmemx_int16_put_warp(&int16_var, &const_int16_t_var, size_var, int_var);
  nvshmemx_int32_put_warp(&int32_var, &const_int32_t_var, size_var, int_var);
  nvshmemx_int64_put_warp(&int64_var, &const_int64_t_var, size_var, int_var);
  nvshmemx_uint8_put_warp(&uint8_var, &const_uint8_t_var, size_var, int_var);
  nvshmemx_uint16_put_warp(&uint16_var, &const_uint16_t_var, size_var, int_var);
  nvshmemx_uint32_put_warp(&uint32_var, &const_uint32_t_var, size_var, int_var);
  nvshmemx_uint64_put_warp(&uint64_var, &const_uint64_t_var, size_var, int_var);
  nvshmemx_size_put_warp(&size_var, &const_size_t_var, size_var, int_var);
  nvshmemx_ptrdiff_put_warp(&ptrdiff_var, &const_ptrdiff_t_var, size_var, int_var);

  // nvshmemx_TYPENAME_wait_on_stream
  // nvshmemx_short_wait_on_stream(&short_var, short_var, cudaStream_t_var);
  // nvshmemx_int_wait_on_stream(&int_var, int_var, cudaStream_t_var);
  // nvshmemx_long_wait_on_stream(&long_var, long_var, cudaStream_t_var);
  // nvshmemx_longlong_wait_on_stream(&longlong_var, longlong_var, cudaStream_t_var);
  // nvshmemx_ushort_wait_on_stream(&ushort_var, ushort_var, cudaStream_t_var);
  // nvshmemx_uint_wait_on_stream(&uint_var, uint_var, cudaStream_t_var);
  // nvshmemx_ulong_wait_on_stream(&ulong_var, ulong_var, cudaStream_t_var);
  // nvshmemx_ulonglong_wait_on_stream(&ulonglong_var, ulonglong_var, cudaStream_t_var);
  // nvshmemx_int32_wait_on_stream(&int32_var, int32_var, cudaStream_t_var);
  // nvshmemx_int64_wait_on_stream(&int64_var, int64_var, cudaStream_t_var);
  // nvshmemx_uint32_wait_on_stream(&uint32_var, uint32_var, cudaStream_t_var);
  // nvshmemx_uint64_wait_on_stream(&uint64_var, uint64_var, cudaStream_t_var);
  // nvshmemx_size_wait_on_stream(&size_var, size_var, cudaStream_t_var);
  // nvshmemx_ptrdiff_wait_on_stream(&ptrdiff_var, ptrdiff_var, cudaStream_t_var);

  nvshmemx_barrier_all_block();
  nvshmemx_barrier_all_warp();
  nvshmemx_barrier_block(nvshmem_team_t_var);
  nvshmemx_barrier_warp(nvshmem_team_t_var);

  // nvshmemx_getSIZE_block (8, 16, 32, 64, 128)
  nvshmemx_get8_block(void_ptr_var, const_void_var, size_var, int_var);
  nvshmemx_get16_block(void_ptr_var, const_void_var, size_var, int_var);
  nvshmemx_get32_block(void_ptr_var, const_void_var, size_var, int_var);
  nvshmemx_get64_block(void_ptr_var, const_void_var, size_var, int_var);
  nvshmemx_get128_block(void_ptr_var, const_void_var, size_var, int_var);

  // nvshmemx_getSIZE_nbi_block (8, 16, 32, 64, 128)
  nvshmemx_get8_nbi_block(void_ptr_var, const_void_var, size_var, int_var);
  nvshmemx_get16_nbi_block(void_ptr_var, const_void_var, size_var, int_var);
  nvshmemx_get32_nbi_block(void_ptr_var, const_void_var, size_var, int_var);
  nvshmemx_get64_nbi_block(void_ptr_var, const_void_var, size_var, int_var);
  nvshmemx_get128_nbi_block(void_ptr_var, const_void_var, size_var, int_var);

  // nvshmemx_getSIZE_nbi_warp (8, 16, 32, 64, 128)
  nvshmemx_get8_nbi_warp(void_ptr_var, const_void_var, size_var, int_var);
  nvshmemx_get16_nbi_warp(void_ptr_var, const_void_var, size_var, int_var);
  nvshmemx_get32_nbi_warp(void_ptr_var, const_void_var, size_var, int_var);
  nvshmemx_get64_nbi_warp(void_ptr_var, const_void_var, size_var, int_var);
  nvshmemx_get128_nbi_warp(void_ptr_var, const_void_var, size_var, int_var);

  // nvshmemx_getSIZE_warp (8, 16, 32, 64, 128)
  nvshmemx_get8_warp(void_ptr_var, const_void_var, size_var, int_var);
  nvshmemx_get16_warp(void_ptr_var, const_void_var, size_var, int_var);
  nvshmemx_get32_warp(void_ptr_var, const_void_var, size_var, int_var);
  nvshmemx_get64_warp(void_ptr_var, const_void_var, size_var, int_var);
  nvshmemx_get128_warp(void_ptr_var, const_void_var, size_var, int_var);

  // nvshmemx_getSIZE_block (8, 16, 32, 64, 128)
  nvshmemx_getmem_block(void_ptr_var, const_void_var, size_var, int_var);
  nvshmemx_getmem_nbi_block(void_ptr_var, const_void_var, size_var, int_var);

  nvshmemx_getmem_nbi_warp(void_ptr_var, const_void_var, size_var, int_var);
  nvshmemx_getmem_warp(void_ptr_var, const_void_var, size_var, int_var);

  // nvshmemx_igetSIZE_block (8, 16, 32, 64, 128)
  nvshmemx_iget8_block(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_iget16_block(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_iget32_block(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_iget64_block(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_iget128_block(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var);

  // nvshmemx_igetSIZE_warp (8, 16, 32, 64, 128)
  nvshmemx_iget8_warp(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_iget16_warp(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_iget32_warp(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_iget64_warp(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_iget128_warp(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var);

  // nvshmemx_iputSIZE_block (8, 16, 32, 64, 128)
  nvshmemx_iput8_block(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_iput16_block(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_iput32_block(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_iput64_block(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_iput128_block(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var);

  // nvshmemx_iputSIZE_warp (8, 16, 32, 64, 128)
  nvshmemx_iput8_warp(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_iput16_warp(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_iput32_warp(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_iput64_warp(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var);
  nvshmemx_iput128_warp(void_ptr_var, const_void_var, ptrdiff_var, ptrdiff_var, size_var, int_var);

  // nvshmemx_putSIZE_block (8, 16, 32, 64, 128)
  nvshmemx_put8_block(void_ptr_var, const_void_var, size_var, int_var);
  nvshmemx_put16_block(void_ptr_var, const_void_var, size_var, int_var);
  nvshmemx_put32_block(void_ptr_var, const_void_var, size_var, int_var);
  nvshmemx_put64_block(void_ptr_var, const_void_var, size_var, int_var);
  nvshmemx_put128_block(void_ptr_var, const_void_var, size_var, int_var);

  // nvshmemx_putSIZE_nbi_block (8, 16, 32, 64, 128)
  nvshmemx_put8_nbi_block(void_ptr_var, const_void_var, size_var, int_var);
  nvshmemx_put16_nbi_block(void_ptr_var, const_void_var, size_var, int_var);
  nvshmemx_put32_nbi_block(void_ptr_var, const_void_var, size_var, int_var);
  nvshmemx_put64_nbi_block(void_ptr_var, const_void_var, size_var, int_var);
  nvshmemx_put128_nbi_block(void_ptr_var, const_void_var, size_var, int_var);

  // nvshmemx_putSIZE_nbi_warp (8, 16, 32, 64, 128)
  nvshmemx_put8_nbi_warp(void_ptr_var, const_void_var, size_var, int_var);
  nvshmemx_put16_nbi_warp(void_ptr_var, const_void_var, size_var, int_var);
  nvshmemx_put32_nbi_warp(void_ptr_var, const_void_var, size_var, int_var);
  nvshmemx_put64_nbi_warp(void_ptr_var, const_void_var, size_var, int_var);
  nvshmemx_put128_nbi_warp(void_ptr_var, const_void_var, size_var, int_var);

  // nvshmemx_putSIZE_signal_block (8, 16, 32, 64, 128)
  nvshmemx_put8_signal_block(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  // nvshmemx_put16_signal_block(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_put32_signal_block(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_put64_signal_block(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_put128_signal_block(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var);

  // nvshmemx_putSIZE_signal_nbi_block (8, 16, 32, 64, 128)
  nvshmemx_put8_signal_nbi_block(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  // nvshmemx_put16_signal_nbi_block(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_put32_signal_nbi_block(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_put64_signal_nbi_block(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_put128_signal_nbi_block(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var);

  // nvshmemx_putSIZE_signal_nbi_warp (8, 16, 32, 64, 128)
  nvshmemx_put8_signal_nbi_warp(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  // nvshmemx_put16_signal_nbi_warp(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_put32_signal_nbi_warp(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_put64_signal_nbi_warp(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_put128_signal_nbi_warp(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var);

  // nvshmemx_putSIZE_signal_warp (8, 16, 32, 64, 128)
  nvshmemx_put8_signal_warp(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  // nvshmemx_put16_signal_warp(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_put32_signal_warp(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_put64_signal_warp(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_put128_signal_warp(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var);

  // nvshmemx_putSIZE_warp (8, 16, 32, 64, 128)
  nvshmemx_put8_warp(void_ptr_var, const_void_var, size_var, int_var);
  nvshmemx_put16_warp(void_ptr_var, const_void_var, size_var, int_var);
  nvshmemx_put32_warp(void_ptr_var, const_void_var, size_var, int_var);
  nvshmemx_put64_warp(void_ptr_var, const_void_var, size_var, int_var);
  nvshmemx_put128_warp(void_ptr_var, const_void_var, size_var, int_var);

  nvshmemx_putmem_block(void_ptr_var, const_void_var, size_var, int_var);

  nvshmemx_putmem_nbi_block(void_ptr_var, const_void_var, size_var, int_var);
  nvshmemx_putmem_nbi_warp(void_ptr_var, const_void_var, size_var, int_var);

  nvshmemx_putmem_signal_block(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_putmem_signal_nbi_block(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_putmem_signal_nbi_warp(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var);
  nvshmemx_putmem_signal_warp(void_ptr_var, const_void_var, size_var, &uint64_var, uint64_var, int_var, int_var);

  nvshmemx_putmem_warp(void_ptr_var, const_void_var, size_var, int_var);

  nvshmemx_sync_all_block();
  nvshmemx_sync_all_warp();
}

int main() {
  h_test();
  d_test();

  return 0;
}
