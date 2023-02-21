// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none --assume-nd-range-dim=1  -out-root %T/cu_jit %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file %T/cu_jit/cu_jit.dp.cpp --match-full-lines %s

#define CU_JIT_NOT_A_CUDA_OPTION 1241

int main() {
  int a[40];
  int CUvar;

//CHECK:a[0] = {{[0-9]+}};
//CHECK-NEXT:a[1] = {{[0-9]+}};
//CHECK-NEXT:a[2] = {{[0-9]+}};
//CHECK-NEXT:a[3] = {{[0-9]+}};
//CHECK-NEXT:a[4] = {{[0-9]+}};
//CHECK-NEXT:a[5] = {{[0-9]+}};
//CHECK-NEXT:a[6] = {{[0-9]+}};
//CHECK-NEXT:a[7] = {{[0-9]+}};
//CHECK-NEXT:a[8] = {{[0-9]+}};
//CHECK-NEXT:a[9] = {{[0-9]+}};
//CHECK-NEXT:a[10] = {{[0-9]+}};
//CHECK-NEXT:a[11] = {{[0-9]+}};
//CHECK-NEXT:a[12] = {{[0-9]+}};
//CHECK-NEXT:a[13] = {{[0-9]+}};
//CHECK-NEXT:a[14] = {{[0-9]+}};
//CHECK-NEXT:a[15] = {{[0-9]+}};
//CHECK-NEXT:a[16] = {{[0-9]+}};
//CHECK-NEXT:a[17] = {{[0-9]+}};
//CHECK-NEXT:a[18] = {{[0-9]+}};
//CHECK-NEXT:a[19] = {{[0-9]+}};
//CHECK-NEXT:a[20] = {{[0-9]+}};
//CHECK-NEXT:a[21] = {{[0-9]+}};
//CHECK-NEXT:a[22] = {{[0-9]+}};
//CHECK-NEXT:a[23] = {{[0-9]+}};
//CHECK-NEXT:a[24] = {{[0-9]+}};
//CHECK-NEXT:a[25] = {{[0-9]+}};
//CHECK-NEXT:a[26] = {{[0-9]+}};
//CHECK-NEXT:a[27] = CU_JIT_NOT_A_CUDA_OPTION;

  a[0] = CU_JIT_MAX_REGISTERS;
  a[1] = CU_JIT_THREADS_PER_BLOCK;
  a[2] = CU_JIT_WALL_TIME;
  a[3] = CU_JIT_INFO_LOG_BUFFER;
  a[4] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
  a[5] = CU_JIT_ERROR_LOG_BUFFER;
  a[6] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
  a[7] = CU_JIT_OPTIMIZATION_LEVEL;
  a[8] = CU_JIT_TARGET_FROM_CUCONTEXT;
  a[9] = CU_JIT_TARGET;
  a[10] = CU_JIT_FALLBACK_STRATEGY;
  a[11] = CU_JIT_GENERATE_DEBUG_INFO;
  a[12] = CU_JIT_LOG_VERBOSE;
  a[13] = CU_JIT_GENERATE_LINE_INFO;
  a[14] = CU_JIT_CACHE_MODE;
  a[15] = CU_JIT_NEW_SM3X_OPT;
  a[16] = CU_JIT_FAST_COMPILE;
  a[17] = CU_JIT_NUM_OPTIONS;
  a[18] = CU_JIT_CACHE_OPTION_NONE;
  a[19] = CU_JIT_CACHE_OPTION_CG;
  a[20] = CU_JIT_CACHE_OPTION_CA;
  a[21] = CU_JIT_INPUT_CUBIN;
  a[22] = CU_JIT_INPUT_PTX;
  a[23] = CU_JIT_INPUT_FATBINARY;
  a[24] = CU_JIT_INPUT_OBJECT;
  a[25] = CU_JIT_INPUT_LIBRARY;
  a[26] = CU_JIT_NUM_INPUT_TYPES;
  a[27] = CU_JIT_NOT_A_CUDA_OPTION;
}