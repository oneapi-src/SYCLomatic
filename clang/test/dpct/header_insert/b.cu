// UNSUPPORTED: system-windows
// RUN: cd %T
// RUN: cp %S/b.cu .
// RUN: cp %S/b.cpp .
// RUN: cp %S/b.h .
// RUN: echo "[" > compile_commands.json
// RUN: echo "    {" >> compile_commands.json
// RUN: echo "        \"command\": \"c++ b.cpp\"," >> compile_commands.json
// RUN: echo "        \"directory\": \"%/T\"," >> compile_commands.json
// RUN: echo "        \"file\": \"%/T/b.cpp\"" >> compile_commands.json
// RUN: echo "    }," >> compile_commands.json
// RUN: echo "    {" >> compile_commands.json
// RUN: echo "        \"command\": \"nvcc b.cu\"," >> compile_commands.json
// RUN: echo "        \"directory\": \"%/T\"," >> compile_commands.json
// RUN: echo "        \"file\": \"%/T/b.cu\"" >> compile_commands.json
// RUN: echo "    }" >> compile_commands.json
// RUN: echo "]" >> compile_commands.json
// RUN: dpct -p=. --out-root=./out --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %S/b.cu  --match-full-lines --input-file %T/out/b.dp.cpp
// RUN: FileCheck %S/b.cpp --match-full-lines --input-file %T/out/b.cpp
// RUN: FileCheck %S/b.h   --match-full-lines --input-file %T/out/b.h
// RUN: %if build_lit %{icpx -c -fsycl %T/out/b.dp.cpp -o %T/out/b.dp.o %}
// RUN: %if build_lit %{icpx -c -fsycl %T/out/b.cpp -o %T/out/b.o %}
// RUN: rm -rf ./out

// CHECK: #define CU_FILE
// CHECK-NEXT: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include "b.h"
// CHECK-EMPTY:
// CHECK-NEXT: void foo() {
// CHECK-NEXT:   sycl::float2 f2;
// CHECK-NEXT: }

#define CU_FILE
#include "b.h"

void foo() {
  float2 f2;
}
