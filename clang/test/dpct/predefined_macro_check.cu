// RUN: dpct --format-range=none  -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/predefined_macro_check.dp.cpp --match-full-lines %s

//CHECK:#ifdef DPCPP_COMPATIBILITY_TEMP
#ifdef __NVCC__
void fun(){
 }
#else
@error  "error"
#endif
