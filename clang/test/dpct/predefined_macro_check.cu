// RUN: dpct --format-range=none -out-root %T/predefined_macro_check %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/predefined_macro_check/predefined_macro_check.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/predefined_macro_check/predefined_macro_check.dp.cpp -o %T/predefined_macro_check/predefined_macro_check.dp.o %}

//CHECK:#ifdef DPCT_COMPATIBILITY_TEMP
#ifdef __NVCC__
void fun(){
 }
#else
@error  "error"
#endif

