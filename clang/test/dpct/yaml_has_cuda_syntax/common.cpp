// UNSUPPORTED: system-windows
// RUN: echo "[" > %T/compile_commands.json
// RUN: echo "    {" >> %T/compile_commands.json
// RUN: echo "        \"command\": \"nvcc %S/common.cpp\"," >> %T/compile_commands.json
// RUN: echo "        \"directory\": \"%T\"," >> %T/compile_commands.json
// RUN: echo "        \"file\": \"%S/common.cpp\"" >> %T/compile_commands.json
// RUN: echo "    }," >> %T/compile_commands.json
// RUN: echo "    {" >> %T/compile_commands.json
// RUN: echo "        \"command\": \"nvcc %S/utils.cpp\"," >> %T/compile_commands.json
// RUN: echo "        \"directory\": \"%T\"," >> %T/compile_commands.json
// RUN: echo "        \"file\": \"%S/utils.cpp\"" >> %T/compile_commands.json
// RUN: echo "    }" >> %T/compile_commands.json
// RUN: echo "]" >> %T/compile_commands.json

// RUN: dpct -in-root=%S -p=%T --out-root=%T/out --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/out/common.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/out/common.cpp -o %T/out/common.o %}

// RUN: FileCheck %S/utils.cpp --match-full-lines --input-file %T/out/utils.cpp.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/out/utils.cpp.dp.cpp -o %T/out/utils.cpp.dp.o %}

// RUN: cat %S/MainSourceFiles.ref > %T/out/check_MainSourceFiles.yaml
// RUN: cat %T/out/MainSourceFiles.yaml >>%T/out/check_MainSourceFiles.yaml
// RUN: FileCheck --match-full-lines --input-file %T/out/check_MainSourceFiles.yaml %T/out/check_MainSourceFiles.yaml

// CHECK: #include <cmath>

void test() {
  int a = -1;
  // CHECK: abs(a);
  abs(a);
}
