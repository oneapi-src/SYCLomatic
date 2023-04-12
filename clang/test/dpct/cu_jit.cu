// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none --assume-nd-range-dim=1  -out-root %T/cu_jit %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file %T/cu_jit/cu_jit.dp.cpp --match-full-lines %s

#include "cuda.h"
#define CU_JIT_NOT_A_CUDA_OPTION 1241

int main() {
  int a[40];
  int CUvar;

//CHECK:DPCT1048:{{[0-9]+}}: The original value CU_JIT_MAX_REGISTERS is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
//CHECK:a[0] = 0;
  a[0] = CU_JIT_MAX_REGISTERS;

//CHECK:DPCT1048:{{[0-9]+}}: The original value CU_JIT_THREADS_PER_BLOCK is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
//CHECK:a[1] = 0;
  a[1] = CU_JIT_THREADS_PER_BLOCK;

//CHECK:DPCT1048:{{[0-9]+}}: The original value CU_JIT_WALL_TIME is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
//CHECK:a[2] = 0;
  a[2] = CU_JIT_WALL_TIME;

//CHECK:DPCT1048:{{[0-9]+}}: The original value CU_JIT_INFO_LOG_BUFFER is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
//CHECK:a[3] = 0;
  a[3] = CU_JIT_INFO_LOG_BUFFER;

//CHECK:DPCT1048:{{[0-9]+}}: The original value CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
//CHECK:a[4] = 0;
  a[4] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;

//CHECK:DPCT1048:{{[0-9]+}}: The original value CU_JIT_ERROR_LOG_BUFFER is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
//CHECK:a[5] = 0;
  a[5] = CU_JIT_ERROR_LOG_BUFFER;

//CHECK:DPCT1048:{{[0-9]+}}: The original value CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
//CHECK:a[6] = 0;
  a[6] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;

//CHECK:DPCT1048:{{[0-9]+}}: The original value CU_JIT_OPTIMIZATION_LEVEL is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
//CHECK:a[7] = 0;
  a[7] = CU_JIT_OPTIMIZATION_LEVEL;

//CHECK:DPCT1048:{{[0-9]+}}: The original value CU_JIT_TARGET_FROM_CUCONTEXT is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
//CHECK:a[8] = 0;
  a[8] = CU_JIT_TARGET_FROM_CUCONTEXT;

//CHECK:DPCT1048:{{[0-9]+}}: The original value CU_JIT_TARGET is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
//CHECK:a[9] = 0;
  a[9] = CU_JIT_TARGET;

//CHECK:DPCT1048:{{[0-9]+}}: The original value CU_JIT_FALLBACK_STRATEGY is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
//CHECK:a[10] = 0;
  a[10] = CU_JIT_FALLBACK_STRATEGY;

//CHECK:DPCT1048:{{[0-9]+}}: The original value CU_JIT_GENERATE_DEBUG_INFO is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
//CHECK:a[11] = 0;
  a[11] = CU_JIT_GENERATE_DEBUG_INFO;

//CHECK:DPCT1048:{{[0-9]+}}: The original value CU_JIT_LOG_VERBOSE is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
//CHECK:a[12] = 0;
  a[12] = CU_JIT_LOG_VERBOSE;

//CHECK:DPCT1048:{{[0-9]+}}: The original value CU_JIT_GENERATE_LINE_INFO is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
//CHECK:a[13] = 0;
  a[13] = CU_JIT_GENERATE_LINE_INFO;

//CHECK:DPCT1048:{{[0-9]+}}: The original value CU_JIT_CACHE_MODE is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
//CHECK:a[14] = 0;
  a[14] = CU_JIT_CACHE_MODE;

//CHECK:DPCT1048:{{[0-9]+}}: The original value CU_JIT_NEW_SM3X_OPT is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
//CHECK:a[15] = 0;
  a[15] = CU_JIT_NEW_SM3X_OPT;

//CHECK:DPCT1048:{{[0-9]+}}: The original value CU_JIT_FAST_COMPILE is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
//CHECK:a[16] = 0;
  a[16] = CU_JIT_FAST_COMPILE;

//CHECK:DPCT1048:{{[0-9]+}}: The original value CU_JIT_NUM_OPTIONS is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
//CHECK:a[17] = 0;
  a[17] = CU_JIT_NUM_OPTIONS;

//CHECK:DPCT1048:{{[0-9]+}}: The original value CU_JIT_CACHE_OPTION_NONE is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
//CHECK:a[18] = 0;
  a[18] = CU_JIT_CACHE_OPTION_NONE;

//CHECK:DPCT1048:{{[0-9]+}}: The original value CU_JIT_CACHE_OPTION_CG is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
//CHECK:a[19] = 0;
  a[19] = CU_JIT_CACHE_OPTION_CG;

//CHECK:DPCT1048:{{[0-9]+}}: The original value CU_JIT_CACHE_OPTION_CA is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
//CHECK:a[20] = 0;
  a[20] = CU_JIT_CACHE_OPTION_CA;

//CHECK:DPCT1048:{{[0-9]+}}: The original value CU_JIT_INPUT_CUBIN is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
//CHECK:a[21] = 0;
  a[21] = CU_JIT_INPUT_CUBIN;

//CHECK:DPCT1048:{{[0-9]+}}: The original value CU_JIT_INPUT_PTX is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
//CHECK:a[22] = 0;
  a[22] = CU_JIT_INPUT_PTX;

//CHECK:DPCT1048:{{[0-9]+}}: The original value CU_JIT_INPUT_FATBINARY is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
//CHECK:a[23] = 0;
  a[23] = CU_JIT_INPUT_FATBINARY;

//CHECK:DPCT1048:{{[0-9]+}}: The original value CU_JIT_INPUT_OBJECT is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
//CHECK:a[24] = 0;
  a[24] = CU_JIT_INPUT_OBJECT;

//CHECK:DPCT1048:{{[0-9]+}}: The original value CU_JIT_INPUT_LIBRARY is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
//CHECK:a[25] = 0;
  a[25] = CU_JIT_INPUT_LIBRARY;

//CHECK:DPCT1048:{{[0-9]+}}: The original value CU_JIT_NUM_INPUT_TYPES is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
//CHECK:a[26] = 0;
  a[26] = CU_JIT_NUM_INPUT_TYPES;

//CHECK:a[27] = CU_JIT_NOT_A_CUDA_OPTION;
  a[27] = CU_JIT_NOT_A_CUDA_OPTION;
}
