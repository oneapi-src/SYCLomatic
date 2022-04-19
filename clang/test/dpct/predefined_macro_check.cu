// RUN: c2s --format-range=none -out-root %T/predefined_macro_check %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/predefined_macro_check/predefined_macro_check.dp.cpp --match-full-lines %s

//CHECK:#ifdef C2S_COMPATIBILITY_TEMP
#ifdef __NVCC__
void fun(){
 }
#else
@error  "error"
#endif

