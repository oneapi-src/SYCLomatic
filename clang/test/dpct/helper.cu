// RUN: dpct --process-all -in-root %S --format-range=none -out-root %T/helper %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --match-full-lines --input-file %T/helper/helper.dp.cpp %s
// RUN: %if build_lit %{icpx -c -fsycl %T/helper/helper.dp.cpp -o %T/helper/helper.dp.o %}

// CHECK: #pragma once
// CHECK-NEXT: #include <stdio.h>
#pragma once
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// CHECK: typedef struct dpct_type_{{[a-f0-9]+}} {
// CHECK-NEXT:    int val;
// CHECK-NEXT: } Pointer;
typedef struct {
   int val;
} Pointer;

#ifdef __cplusplus
}
#endif

