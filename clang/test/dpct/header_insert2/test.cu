// UNSUPPORTED: system-windows
// RUN: cd %T
// RUN: cp %S/test.cu .
// RUN: cp %S/test.cpp .
// RUN: cp %S/test.h .
// RUN: echo "[" > compile_commands.json
// RUN: echo "    {" >> compile_commands.json
// RUN: echo "        \"command\": \"c++ test.cpp\"," >> compile_commands.json
// RUN: echo "        \"directory\": \"%/T\"," >> compile_commands.json
// RUN: echo "        \"file\": \"%/T/test.cpp\"" >> compile_commands.json
// RUN: echo "    }," >> compile_commands.json
// RUN: echo "    {" >> compile_commands.json
// RUN: echo "        \"command\": \"nvcc test.cu\"," >> compile_commands.json
// RUN: echo "        \"directory\": \"%/T\"," >> compile_commands.json
// RUN: echo "        \"file\": \"%/T/test.cu\"" >> compile_commands.json
// RUN: echo "    }" >> compile_commands.json
// RUN: echo "]" >> compile_commands.json
// RUN: dpct -p=. --out-root=./out --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %S/test.cu  --match-full-lines --input-file %T/out/test.dp.cpp
// RUN: FileCheck %S/test.cpp --match-full-lines --input-file %T/out/test.cpp
// RUN: FileCheck %S/test.h   --match-full-lines --input-file %T/out/test.h
// RUN: rm -rf ./out

// CHECK: #define CU_FILE
// CHECK-NEXT: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include "test.h"
// CHECK-EMPTY:
// CHECK-NEXT: void foo() {
// CHECK-NEXT:   sycl::float2 f2;
// CHECK-NEXT: }

#define CU_FILE
#include "test.h"

void foo() {
  float2 f2;
}
