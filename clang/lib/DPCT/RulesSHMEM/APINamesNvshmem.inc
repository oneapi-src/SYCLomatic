//===------------------------ APINamesNvshmem.inc -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Helper macros for stringizing
#define STRINGIZE(x) #x
#define EXPAND_AND_STRINGIZE(x) STRINGIZE(x)

// Helper function for registering API rules for starndard RMA data types
#define FOR_ALL_STANDARD_RMA_TYPES(PREFIX, MIGRATION_RULE, OP, N_POSTFIX,      \
                                   I_POSTFIX)                                  \
  MIGRATION_RULE(PREFIX, OP, float, N_POSTFIX, I_POSTFIX)                      \
  MIGRATION_RULE(PREFIX, OP, double, N_POSTFIX, I_POSTFIX)                     \
  MIGRATION_RULE(PREFIX, OP, char, N_POSTFIX, I_POSTFIX)                       \
  MIGRATION_RULE(PREFIX, OP, short, N_POSTFIX, I_POSTFIX)                      \
  MIGRATION_RULE(PREFIX, OP, schar, N_POSTFIX, I_POSTFIX)                      \
  MIGRATION_RULE(PREFIX, OP, int, N_POSTFIX, I_POSTFIX)                        \
  MIGRATION_RULE(PREFIX, OP, long, N_POSTFIX, I_POSTFIX)                       \
  MIGRATION_RULE(PREFIX, OP, longlong, N_POSTFIX, I_POSTFIX)                   \
  MIGRATION_RULE(PREFIX, OP, uchar, N_POSTFIX, I_POSTFIX)                      \
  MIGRATION_RULE(PREFIX, OP, ushort, N_POSTFIX, I_POSTFIX)                     \
  MIGRATION_RULE(PREFIX, OP, uint, N_POSTFIX, I_POSTFIX)                       \
  MIGRATION_RULE(PREFIX, OP, ulong, N_POSTFIX, I_POSTFIX)                      \
  MIGRATION_RULE(PREFIX, OP, ulonglong, N_POSTFIX, I_POSTFIX)                  \
  MIGRATION_RULE(PREFIX, OP, int8, N_POSTFIX, I_POSTFIX)                       \
  MIGRATION_RULE(PREFIX, OP, int16, N_POSTFIX, I_POSTFIX)                      \
  MIGRATION_RULE(PREFIX, OP, int32, N_POSTFIX, I_POSTFIX)                      \
  MIGRATION_RULE(PREFIX, OP, int64, N_POSTFIX, I_POSTFIX)                      \
  MIGRATION_RULE(PREFIX, OP, uint8, N_POSTFIX, I_POSTFIX)                      \
  MIGRATION_RULE(PREFIX, OP, uint16, N_POSTFIX, I_POSTFIX)                     \
  MIGRATION_RULE(PREFIX, OP, uint32, N_POSTFIX, I_POSTFIX)                     \
  MIGRATION_RULE(PREFIX, OP, uint64, N_POSTFIX, I_POSTFIX)                     \
  MIGRATION_RULE(PREFIX, OP, size, N_POSTFIX, I_POSTFIX)                       \
  MIGRATION_RULE(PREFIX, OP, ptrdiff, N_POSTFIX, I_POSTFIX)

// Helper function for registering API rules for starndard RMA data types
#define FOR_ALL_SIZES(PREFIX, MIGRATION_RULE, OP, N_POSTFIX, I_POSTFIX)        \
  MIGRATION_RULE(PREFIX, OP, 8, N_POSTFIX, I_POSTFIX)                          \
  MIGRATION_RULE(PREFIX, OP, 16, N_POSTFIX, I_POSTFIX)                         \
  MIGRATION_RULE(PREFIX, OP, 32, N_POSTFIX, I_POSTFIX)                         \
  MIGRATION_RULE(PREFIX, OP, 64, N_POSTFIX, I_POSTFIX)                         \
  MIGRATION_RULE(PREFIX, OP, 128, N_POSTFIX, I_POSTFIX)

// Helper function for registering API rule
#define REGISTER_NVSHMEM_RULE(PREFIX, OP, TYPE, N_POSTFIX, I_POSTFIX)          \
  FEATURE_REQUEST_FACTORY(                                                     \
      HelperFeatureEnum::device_ext,                                           \
      CALL_FACTORY_ENTRY(                                                      \
          EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##TYPE##_##OP##N_POSTFIX),    \
          CALL(EXPAND_AND_STRINGIZE(ishmem##PREFIX##_##OP##I_POSTFIX), ARG(0), \
               ARG(1), ARG(2), ARG(3))))

#define REGISTER_STREAM_NVSHMEM_RULE(PREFIX, OP, TYPE, N_POSTFIX, I_POSTFIX)   \
  FEATURE_REQUEST_FACTORY(                                                     \
      HelperFeatureEnum::device_ext,                                           \
      CALL_FACTORY_ENTRY(                                                      \
          EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##TYPE##_##OP##N_POSTFIX),    \
          CALL(EXPAND_AND_STRINGIZE(ishmem##PREFIX##_##OP##I_POSTFIX), ARG(0), \
               ARG(1), ARG(2), ARG(3), DEREF(4))))

#define REGISTER_BLOCK_NVSHMEM_RULE(PREFIX, OP, TYPE, N_POSTFIX, I_POSTFIX)    \
  FEATURE_REQUEST_FACTORY(                                                     \
      HelperFeatureEnum::device_ext,                                           \
      CALL_FACTORY_ENTRY(                                                      \
          EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##TYPE##_##OP##N_POSTFIX),    \
          CALL(EXPAND_AND_STRINGIZE(ishmem##PREFIX##_##OP##I_POSTFIX), ARG(0), \
               ARG(1), ARG(2), ARG(3), GROUP)))

#define REGISTER_WARP_NVSHMEM_RULE(PREFIX, OP, TYPE, N_POSTFIX, I_POSTFIX)     \
  FEATURE_REQUEST_FACTORY(                                                     \
      HelperFeatureEnum::device_ext,                                           \
      CALL_FACTORY_ENTRY(                                                      \
          EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##TYPE##_##OP##N_POSTFIX),    \
          CALL(EXPAND_AND_STRINGIZE(ishmem##PREFIX##_##OP##I_POSTFIX), ARG(0), \
               ARG(1), ARG(2), ARG(3), SUBGROUP)))

#define REGISTER_SIZE_NVSHMEM_RULE(PREFIX, OP, SIZE, N_POSTFIX, I_POSTFIX)     \
  FEATURE_REQUEST_FACTORY(                                                     \
      HelperFeatureEnum::device_ext,                                           \
      CALL_FACTORY_ENTRY(                                                      \
          EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##OP##SIZE##N_POSTFIX),       \
          CALL(EXPAND_AND_STRINGIZE(ishmem##PREFIX##_##OP##SIZE##I_POSTFIX),   \
               ARG(0), ARG(1), ARG(2), ARG(3))))

#define REGISTER_SIZE_STREAM_NVSHMEM_RULE(PREFIX, OP, SIZE, N_POSTFIX,         \
                                          I_POSTFIX)                           \
  FEATURE_REQUEST_FACTORY(                                                     \
      HelperFeatureEnum::device_ext,                                           \
      CALL_FACTORY_ENTRY(                                                      \
          EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##OP##SIZE##N_POSTFIX),       \
          CALL(EXPAND_AND_STRINGIZE(ishmem##PREFIX##_##OP##SIZE##I_POSTFIX),   \
               ARG(0), ARG(1), ARG(2), ARG(3), DEREF(4))))

#define REGISTER_SIZE_BLOCK_NVSHMEM_RULE(PREFIX, OP, SIZE, N_POSTFIX,          \
                                         I_POSTFIX)                            \
  FEATURE_REQUEST_FACTORY(                                                     \
      HelperFeatureEnum::device_ext,                                           \
      CALL_FACTORY_ENTRY(                                                      \
          EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##OP##SIZE##N_POSTFIX),       \
          CALL(EXPAND_AND_STRINGIZE(ishmem##PREFIX##_##OP##SIZE##I_POSTFIX),   \
               ARG(0), ARG(1), ARG(2), ARG(3), GROUP)))

#define REGISTER_SIZE_WARP_NVSHMEM_RULE(PREFIX, OP, SIZE, N_POSTFIX,           \
                                        I_POSTFIX)                             \
  FEATURE_REQUEST_FACTORY(                                                     \
      HelperFeatureEnum::device_ext,                                           \
      CALL_FACTORY_ENTRY(                                                      \
          EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##OP##SIZE##N_POSTFIX),       \
          CALL(EXPAND_AND_STRINGIZE(ishmem##PREFIX##_##OP##SIZE##I_POSTFIX),   \
               ARG(0), ARG(1), ARG(2), ARG(3), SUBGROUP)))

#define REGISTER_STRIDED_NVSHMEM_RULE(PREFIX, OP, TYPE, N_POSTFIX, I_POSTFIX)  \
  FEATURE_REQUEST_FACTORY(                                                     \
      HelperFeatureEnum::device_ext,                                           \
      CALL_FACTORY_ENTRY(                                                      \
          EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##TYPE##_##OP##N_POSTFIX),    \
          CALL(EXPAND_AND_STRINGIZE(ishmem##PREFIX##_##OP##I_POSTFIX), ARG(0), \
               ARG(1), ARG(2), ARG(3), ARG(4), ARG(5))))

#define REGISTER_STRIDED_STREAM_NVSHMEM_RULE(PREFIX, OP, TYPE, N_POSTFIX,      \
                                             I_POSTFIX)                        \
  FEATURE_REQUEST_FACTORY(                                                     \
      HelperFeatureEnum::device_ext,                                           \
      CALL_FACTORY_ENTRY(                                                      \
          EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##TYPE##_##OP##N_POSTFIX),    \
          CALL(EXPAND_AND_STRINGIZE(ishmem##PREFIX##_##OP##I_POSTFIX), ARG(0), \
               ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), DEREF(6))))

#define REGISTER_STRIDED_BLOCK_NVSHMEM_RULE(PREFIX, OP, TYPE, N_POSTFIX,       \
                                            I_POSTFIX)                         \
  FEATURE_REQUEST_FACTORY(                                                     \
      HelperFeatureEnum::device_ext,                                           \
      CALL_FACTORY_ENTRY(                                                      \
          EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##TYPE##_##OP##N_POSTFIX),    \
          CALL(EXPAND_AND_STRINGIZE(ishmem##PREFIX##_##OP##I_POSTFIX), ARG(0), \
               ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), GROUP)))

#define REGISTER_STRIDED_WARP_NVSHMEM_RULE(PREFIX, OP, TYPE, N_POSTFIX,        \
                                           I_POSTFIX)                          \
  FEATURE_REQUEST_FACTORY(                                                     \
      HelperFeatureEnum::device_ext,                                           \
      CALL_FACTORY_ENTRY(                                                      \
          EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##TYPE##_##OP##N_POSTFIX),    \
          CALL(EXPAND_AND_STRINGIZE(ishmem##PREFIX##_##OP##I_POSTFIX), ARG(0), \
               ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), SUBGROUP)))

#define REGISTER_STRIDED_SIZE_NVSHMEM_RULE(PREFIX, OP, SIZE, N_POSTFIX,        \
                                           I_POSTFIX)                          \
  FEATURE_REQUEST_FACTORY(                                                     \
      HelperFeatureEnum::device_ext,                                           \
      CALL_FACTORY_ENTRY(                                                      \
          EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##OP##SIZE##N_POSTFIX),       \
          CALL(EXPAND_AND_STRINGIZE(ishmem##PREFIX##_##OP##SIZE##I_POSTFIX),   \
               ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5))))

#define REGISTER_STRIDED_SIZE_STREAM_NVSHMEM_RULE(PREFIX, OP, SIZE, N_POSTFIX, \
                                                  I_POSTFIX)                   \
  FEATURE_REQUEST_FACTORY(                                                     \
      HelperFeatureEnum::device_ext,                                           \
      CALL_FACTORY_ENTRY(                                                      \
          EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##OP##SIZE##N_POSTFIX),       \
          CALL(EXPAND_AND_STRINGIZE(ishmem##PREFIX##_##OP##SIZE##I_POSTFIX),   \
               ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), DEREF(6))))

#define REGISTER_STRIDED_SIZE_BLOCK_NVSHMEM_RULE(PREFIX, OP, SIZE, N_POSTFIX,  \
                                                 I_POSTFIX)                    \
  FEATURE_REQUEST_FACTORY(                                                     \
      HelperFeatureEnum::device_ext,                                           \
      CALL_FACTORY_ENTRY(                                                      \
          EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##OP##SIZE##N_POSTFIX),       \
          CALL(EXPAND_AND_STRINGIZE(ishmem##PREFIX##_##OP##SIZE##I_POSTFIX),   \
               ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), GROUP)))

#define REGISTER_STRIDED_SIZE_WARP_NVSHMEM_RULE(PREFIX, OP, SIZE, N_POSTFIX,   \
                                                I_POSTFIX)                     \
  FEATURE_REQUEST_FACTORY(                                                     \
      HelperFeatureEnum::device_ext,                                           \
      CALL_FACTORY_ENTRY(                                                      \
          EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##OP##SIZE##N_POSTFIX),       \
          CALL(EXPAND_AND_STRINGIZE(ishmem##PREFIX##_##OP##SIZE##I_POSTFIX),   \
               ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), SUBGROUP)))

#define REGISTER_P_NVSHMEM_RULE(PREFIX, OP, TYPE, N_POSTFIX, I_POSTFIX)        \
  FEATURE_REQUEST_FACTORY(                                                     \
      HelperFeatureEnum::device_ext,                                           \
      CALL_FACTORY_ENTRY(                                                      \
          EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##TYPE##_##OP##N_POSTFIX),    \
          CALL(EXPAND_AND_STRINGIZE(ishmem##PREFIX##_##OP##I_POSTFIX), ARG(0), \
               ARG(1), ARG(2))))

#define REGISTER_G_NVSHMEM_RULE(PREFIX, OP, TYPE, N_POSTFIX, I_POSTFIX)        \
  FEATURE_REQUEST_FACTORY(                                                     \
      HelperFeatureEnum::device_ext,                                           \
      CALL_FACTORY_ENTRY(                                                      \
          EXPAND_AND_STRINGIZE(nvshmem##PREFIX##_##TYPE##_##OP##N_POSTFIX),    \
          CALL(EXPAND_AND_STRINGIZE(ishmem##PREFIX##_##OP##I_POSTFIX), ARG(0), \
               ARG(1))))

// Library Setup, Exit & Query
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("nvshmem_init", CALL("ishmem_init")))

HEADER_INSERT_FACTORY(HeaderType::HT_DPCT_SHMEM_Utils,
                      FEATURE_REQUEST_FACTORY(
                          HelperFeatureEnum::device_ext,
                          CALL_FACTORY_ENTRY("nvshmemx_init_attr",
                                             CALL(MapNames::getDpctNamespace() +
                                                      "shmemx::init_attr",
                                                  ARG(0), ARG(1)))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("nvshmem_my_pe",
                                           CALL("ishmem_my_pe")))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("nvshmem_n_pes",
                                           CALL("ishmem_n_pes")))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("nvshmem_finalize",
                                           CALL("ishmem_finalize")))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("nvshmem_ptr",
                                           CALL("ishmem_ptr", ARG(0), ARG(1))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("nvshmem_info_get_version",
                                           CALL("ishmem_info_get_version",
                                                ARG(0), ARG(1))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("nvshmem_info_get_name",
                                           CALL("ishmem_info_get_name",
                                                ARG(0))))

// Memory Management
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("nvshmem_malloc",
                                           CALL("ishmem_malloc", ARG(0))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("nvshmem_free",
                                           CALL("ishmem_free", ARG(0))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("nvshmem_align", CALL("ishmem_align", ARG(0), ARG(1))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("nvshmem_calloc", CALL("ishmem_calloc", ARG(0), ARG(1))))

// Team Management
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("nvshmem_team_my_pe",
                                           CALL("ishmem_team_my_pe", ARG(0))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("nvshmem_team_n_pes",
                                           CALL("ishmem_team_n_pes", ARG(0))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("nvshmem_team_get_config",
                                           CALL("ishmem_team_get_config",
                                                ARG(0), LITERAL("0"), ARG(1))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("nvshmem_team_translate_pe",
                                           CALL("ishmem_team_translate_pe",
                                                ARG(0), ARG(1), ARG(2))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("nvshmem_team_split_strided",
                                           CALL("ishmem_team_split_strided",
                                                ARG(0), ARG(1), ARG(2), ARG(3),
                                                ARG(4), ARG(5), ARG(6))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("nvshmem_team_split_2d",
                                           CALL("ishmem_team_split_2d", ARG(0),
                                                ARG(1), ARG(2), ARG(3), ARG(4),
                                                ARG(5), ARG(6), ARG(7))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("nvshmem_team_destroy",
                                           CALL("ishmem_team_destroy", ARG(0))))

// RMA
// nvshmem_TYPENAME_put
FOR_ALL_STANDARD_RMA_TYPES(, REGISTER_NVSHMEM_RULE, put, , )
// nvshmemx_TYPENAME_put_on_stream
FOR_ALL_STANDARD_RMA_TYPES(x, REGISTER_STREAM_NVSHMEM_RULE, put, _on_stream,
                           _on_queue)
// nvshmemx_TYPENAME_put_block
FOR_ALL_STANDARD_RMA_TYPES(x, REGISTER_BLOCK_NVSHMEM_RULE, put, _block,
                           _work_group)
// nvshmemx_TYPENAME_put_warp
FOR_ALL_STANDARD_RMA_TYPES(x, REGISTER_WARP_NVSHMEM_RULE, put, _warp,
                           _work_group)

// nvshmem_putSIZE
FOR_ALL_SIZES(, REGISTER_SIZE_NVSHMEM_RULE, put, , )
// nvshmemx_putSIZE_on_stream
FOR_ALL_SIZES(x, REGISTER_SIZE_STREAM_NVSHMEM_RULE, put, _on_stream, _on_queue)
// nvshmem_putSIZE_block
FOR_ALL_SIZES(x, REGISTER_SIZE_BLOCK_NVSHMEM_RULE, put, _block, _work_group)
// nvshmem_putSIZE_warp
FOR_ALL_SIZES(x, REGISTER_SIZE_WARP_NVSHMEM_RULE, put, _warp, _work_group)

// nvshmem_TYPENAME_iput
FOR_ALL_STANDARD_RMA_TYPES(, REGISTER_STRIDED_NVSHMEM_RULE, iput, , )
// nvshmemx_TYPENAME_iput_on_stream
FOR_ALL_STANDARD_RMA_TYPES(x, REGISTER_STRIDED_STREAM_NVSHMEM_RULE, iput,
                           _on_stream, _on_queue)
// nvshmemx_TYPENAME_iput_block
FOR_ALL_STANDARD_RMA_TYPES(x, REGISTER_STRIDED_BLOCK_NVSHMEM_RULE, iput, _block,
                           _work_group)
// nvshmemx_TYPENAME_iput_warp
FOR_ALL_STANDARD_RMA_TYPES(x, REGISTER_STRIDED_WARP_NVSHMEM_RULE, iput, _warp,
                           _work_group)

// nvshmem_iputSIZE
FOR_ALL_SIZES(, REGISTER_STRIDED_SIZE_NVSHMEM_RULE, iput, , )
// nvshmem_iputSIZE_on_stream
FOR_ALL_SIZES(x, REGISTER_STRIDED_SIZE_STREAM_NVSHMEM_RULE, iput, _on_stream,
              _on_queue)
// nvshmem_iputSIZE_block
FOR_ALL_SIZES(x, REGISTER_STRIDED_SIZE_BLOCK_NVSHMEM_RULE, iput, _block,
              _work_group)
// nvshmem_iputSIZE_warp
FOR_ALL_SIZES(x, REGISTER_STRIDED_SIZE_WARP_NVSHMEM_RULE, iput, _warp,
              _work_group)

// nvshmem_putmem
REGISTER_SIZE_NVSHMEM_RULE(, put, mem, , )
// nvshmem_putmem_on_stream
REGISTER_SIZE_STREAM_NVSHMEM_RULE(x, put, mem, _on_stream, _on_queue)
// nvshmem_putmem_block
REGISTER_SIZE_BLOCK_NVSHMEM_RULE(x, put, mem, _block, _work_group)
// nvshmem_putmem_warp
REGISTER_SIZE_WARP_NVSHMEM_RULE(x, put, mem, _warp, _work_group)

// nvshmem_TYPENAME_p
FOR_ALL_STANDARD_RMA_TYPES(, REGISTER_P_NVSHMEM_RULE, p, , )

// nvshmem_TYPENAME_get
FOR_ALL_STANDARD_RMA_TYPES(, REGISTER_NVSHMEM_RULE, get, , )
// nvshmemx_TYPENAME_get_on_stream
FOR_ALL_STANDARD_RMA_TYPES(x, REGISTER_STREAM_NVSHMEM_RULE, get, _on_stream,
                           _on_queue)
// nvshmemx_TYPENAME_get_block
FOR_ALL_STANDARD_RMA_TYPES(x, REGISTER_BLOCK_NVSHMEM_RULE, get, _block,
                           _work_group)
// nvshmemx_TYPENAME_get_warp
FOR_ALL_STANDARD_RMA_TYPES(x, REGISTER_WARP_NVSHMEM_RULE, get, _warp,
                           _work_group)

// nvshmem_getSIZE
FOR_ALL_SIZES(, REGISTER_SIZE_NVSHMEM_RULE, get, , )
// nvshmem_getSIZE_on_stream
FOR_ALL_SIZES(x, REGISTER_SIZE_STREAM_NVSHMEM_RULE, get, _on_stream, _on_queue)
// nvshmem_getSIZE_block
FOR_ALL_SIZES(x, REGISTER_SIZE_BLOCK_NVSHMEM_RULE, get, _block, _work_group)
// nvshmem_getSIZE_warp
FOR_ALL_SIZES(x, REGISTER_SIZE_WARP_NVSHMEM_RULE, get, _warp, _work_group)

// nvshmem_TYPENAME_iget
FOR_ALL_STANDARD_RMA_TYPES(, REGISTER_STRIDED_NVSHMEM_RULE, iget, , )
// nvshmemx_TYPENAME_iget_on_stream
FOR_ALL_STANDARD_RMA_TYPES(x, REGISTER_STRIDED_STREAM_NVSHMEM_RULE, iget,
                           _on_stream, _on_queue)
// nvshmemx_TYPENAME_iget_block
FOR_ALL_STANDARD_RMA_TYPES(x, REGISTER_STRIDED_BLOCK_NVSHMEM_RULE, iget, _block,
                           _work_group)
// nvshmemx_TYPENAME_iget_warp
FOR_ALL_STANDARD_RMA_TYPES(x, REGISTER_STRIDED_WARP_NVSHMEM_RULE, iget, _warp,
                           _work_group)

// nvshmem_igetSIZE
FOR_ALL_SIZES(, REGISTER_STRIDED_SIZE_NVSHMEM_RULE, iget, , )
// nvshmem_igetSIZE_on_stream
FOR_ALL_SIZES(x, REGISTER_STRIDED_SIZE_STREAM_NVSHMEM_RULE, iget, _on_stream,
              _on_queue)
// nvshmem_igetSIZE_block
FOR_ALL_SIZES(x, REGISTER_STRIDED_SIZE_BLOCK_NVSHMEM_RULE, iget, _block,
              _work_group)
// nvshmem_igetSIZE_warp
FOR_ALL_SIZES(x, REGISTER_STRIDED_SIZE_WARP_NVSHMEM_RULE, iget, _warp,
              _work_group)

// nvshmem_getmem
REGISTER_SIZE_NVSHMEM_RULE(, get, mem, , )
// nvshmem_getmem_on_stream
REGISTER_SIZE_STREAM_NVSHMEM_RULE(x, get, mem, _on_stream, _on_queue)
// nvshmem_getmem_block
REGISTER_SIZE_BLOCK_NVSHMEM_RULE(x, get, mem, _block, _work_group)
// nvshmem_getmem_warp
REGISTER_SIZE_WARP_NVSHMEM_RULE(x, get, mem, _warp, _work_group)

// nvshmem_TYPENAME_g
FOR_ALL_STANDARD_RMA_TYPES(, REGISTER_G_NVSHMEM_RULE, g, , )

// Nonblocing RMA
// nvshmem_TYPENAME_put_nbi
FOR_ALL_STANDARD_RMA_TYPES(, REGISTER_NVSHMEM_RULE, put, _nbi, _nbi)
// nvshmemx_TYPENAME_put_nbi_on_stream
FOR_ALL_STANDARD_RMA_TYPES(x, REGISTER_STREAM_NVSHMEM_RULE, put, _nbi_on_stream,
                           _nbi_on_queue)
// nvshmemx_TYPENAME_put_nbi_block
FOR_ALL_STANDARD_RMA_TYPES(x, REGISTER_BLOCK_NVSHMEM_RULE, put, _nbi_block,
                           _nbi_work_group)
// nvshmemx_TYPENAME_put_nbi_warp
FOR_ALL_STANDARD_RMA_TYPES(x, REGISTER_WARP_NVSHMEM_RULE, put, _nbi_warp,
                           _nbi_work_group)

// nvshmem_putSIZE_nbi
FOR_ALL_SIZES(, REGISTER_SIZE_NVSHMEM_RULE, put, _nbi, _nbi)
// nvshmemx_putSIZE_nbi_on_stream
FOR_ALL_SIZES(x, REGISTER_SIZE_STREAM_NVSHMEM_RULE, put, _nbi_on_stream,
              _nbi_on_queue)
// nvshmem_putSIZE_nbi_block
FOR_ALL_SIZES(x, REGISTER_SIZE_BLOCK_NVSHMEM_RULE, put, _nbi_block,
              _nbi_work_group)
// nvshmem_putSIZE_nbi_warp
FOR_ALL_SIZES(x, REGISTER_SIZE_WARP_NVSHMEM_RULE, put, _nbi_warp,
              _nbi_work_group)

// nvshmem_putmem_nbi
REGISTER_SIZE_NVSHMEM_RULE(, put, mem, _nbi, _nbi)
// nvshmem_putmem_nbi_on_stream
REGISTER_SIZE_STREAM_NVSHMEM_RULE(x, put, mem, _nbi_on_stream, _nbi_on_queue)
// nvshmem_putmem_nbi_block
REGISTER_SIZE_BLOCK_NVSHMEM_RULE(x, put, mem, _nbi_block, _nbi_work_group)
// nvshmem_putmem_nbi_warp
REGISTER_SIZE_WARP_NVSHMEM_RULE(x, put, mem, _nbi_warp, _nbi_work_group)

// nvshmem_TYPENAME_get_nbi
FOR_ALL_STANDARD_RMA_TYPES(, REGISTER_NVSHMEM_RULE, get, _nbi, _nbi)
// nvshmemx_TYPENAME_get_nbi_on_stream
FOR_ALL_STANDARD_RMA_TYPES(x, REGISTER_STREAM_NVSHMEM_RULE, get, _nbi_on_stream,
                           _nbi_on_queue)
// nvshmemx_TYPENAME_get_nbi_block
FOR_ALL_STANDARD_RMA_TYPES(x, REGISTER_BLOCK_NVSHMEM_RULE, get, _nbi_block,
                           _nbi_work_group)
// nvshmemx_TYPENAME_get_nbi_warp
FOR_ALL_STANDARD_RMA_TYPES(x, REGISTER_WARP_NVSHMEM_RULE, get, _nbi_warp,
                           _nbi_work_group)

// nvshmem_getSIZE_nbi
FOR_ALL_SIZES(, REGISTER_SIZE_NVSHMEM_RULE, get, _nbi, _nbi)
// nvshmemx_getSIZE_nbi_on_stream
FOR_ALL_SIZES(x, REGISTER_SIZE_STREAM_NVSHMEM_RULE, get, _nbi_on_stream,
              _nbi_on_queue)
// nvshmem_getSIZE_nbi_block
FOR_ALL_SIZES(x, REGISTER_SIZE_BLOCK_NVSHMEM_RULE, get, _nbi_block,
              _nbi_work_group)
// nvshmem_getSIZE_nbi_warp
FOR_ALL_SIZES(x, REGISTER_SIZE_WARP_NVSHMEM_RULE, get, _nbi_warp,
              _nbi_work_group)

// nvshmem_getmem_nbi
REGISTER_SIZE_NVSHMEM_RULE(, get, mem, _nbi, _nbi)
// nvshmem_getmem_nbi_on_stream
REGISTER_SIZE_STREAM_NVSHMEM_RULE(x, get, mem, _nbi_on_stream, _nbi_on_queue)
// nvshmem_getmem_nbi_block
REGISTER_SIZE_BLOCK_NVSHMEM_RULE(x, get, mem, _nbi_block, _nbi_work_group)
// nvshmem_getmem_nbi_warp
REGISTER_SIZE_WARP_NVSHMEM_RULE(x, get, mem, _nbi_warp, _nbi_work_group)

// Memory Ordering
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("nvshmem_fence",
                                           CALL("ishmem_fence")))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("nvshmem_quiet",
                                           CALL("ishmem_quiet")))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("nvshmemx_quiet_on_stream",
                                           CALL("ishmemx_quiet_on_queue",
                                                DEREF(0))))

// Collective Operations
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("nvshmemx_barrier_all_on_stream",
                                           CALL("ishmemx_barrier_all_on_queue",
                                                DEREF(0))))

// Signalling Operations
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("nvshmem_signal_wait_until",
                                           CALL("ishmem_signal_wait_until",
                                                ARG(0), ARG(1), ARG(2))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("nvshmem_putmem_signal_nbi",
                                           CALL("ishmem_putmem_signal_nbi",
                                                ARG(0), ARG(1), ARG(2), ARG(3),
                                                ARG(4), ARG(5), ARG(6))))

HEADER_INSERT_FACTORY(HeaderType::HT_DPCT_SHMEM_Utils,
                      FEATURE_REQUEST_FACTORY(
                          HelperFeatureEnum::device_ext,
                          CALL_FACTORY_ENTRY("nvshmemx_signal_op",
                                             CALL(MapNames::getDpctNamespace() +
                                                      "shmemx::signal_op",
                                                  ARG(0), ARG(1), ARG(2),
                                                  ARG(3)))))

#undef STRINGIZE
#undef EXPAND_AND_STRINGIZE
#undef FOR_ALL_STANDARD_RMA_TYPES
#undef FOR_ALL_SIZES
#undef REGISTER_NVSHMEM_RULE
#undef REGISTER_STREAM_NVSHMEM_RULE
#undef REGISTER_BLOCK_NVSHMEM_RULE
#undef REGISTER_WARP_NVSHMEM_RULE
#undef REGISTER_SIZE_NVSHMEM_RULE
#undef REGISTER_SIZE_STREAM_NVSHMEM_RULE
#undef REGISTER_SIZE_BLOCK_NVSHMEM_RULE
#undef REGISTER_SIZE_WARP_NVSHMEM_RULE
#undef REGISTER_STRIDED_NVSHMEM_RULE
#undef REGISTER_STRIDED_STREAM_NVSHMEM_RULE
#undef REGISTER_STRIDED_BLOCK_NVSHMEM_RULE
#undef REGISTER_STRIDED_WARP_NVSHMEM_RULE
#undef REGISTER_STRIDED_SIZE_NVSHMEM_RULE
#undef REGISTER_STRIDED_SIZE_STREAM_NVSHMEM_RULE
#undef REGISTER_STRIDED_SIZE_BLOCK_NVSHMEM_RULE
#undef REGISTER_STRIDED_SIZE_WARP_NVSHMEM_RULE
#undef REGISTER_P_NVSHMEM_RULE
#undef REGISTER_G_NVSHMEM_RULE
