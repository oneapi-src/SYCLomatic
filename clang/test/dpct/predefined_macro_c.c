// RUN: dpct --format-range=none --out-root %T/predefined_macro_c %s --cuda-include-path="%cuda-path/include" --extra-arg="-xc"
// RUN: FileCheck --input-file %T/predefined_macro_c/predefined_macro_c.c.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/predefined_macro_c/predefined_macro_c.c.dp.cpp -o %T/predefined_macro_c/predefined_macro_c.c.dp.o %}

//CHECK: #define DPCT_COMPAT_RT_VERSION {{[1-9][0-9]+}}

#include <cuda_runtime_api.h>
#include <stdio.h>

//CHECK:#ifdef DPCT_COMPAT_RT_VERSION
//CHECK-NEXT:void hello1() { printf("foo"); }
//CHECK-NEXT:#endif
//CHECK-NEXT:#ifdef DPCT_COMPAT_RT_VERSION
//CHECK-NEXT:void hello2() { printf("foo"); }
//CHECK-NEXT:#endif
#ifdef CUDART_VERSION
void hello1() { printf("foo"); }
#endif
#ifdef __CUDART_API_VERSION
void hello2() { printf("foo"); }
#endif
