// UNSUPPORTED: system-windows
// RUN: cd %T
// RUN: mkdir c_feature_file_db
// RUN: cd c_feature_file_db
// RUN: cat %s > c_feature_file_db.c
// RUN: echo "[" > compile_commands.json
// RUN: echo "    {" >> compile_commands.json
// RUN: echo "        \"command\": \"nvcc c_feature_file_db.c\"," >> compile_commands.json
// RUN: echo "        \"directory\": \"%/T/c_feature_file_db\"," >> compile_commands.json
// RUN: echo "        \"file\": \"%/T/c_feature_file_db/c_feature_file_db.c\"" >> compile_commands.json
// RUN: echo "    }" >> compile_commands.json
// RUN: echo "]" >> compile_commands.json
// RUN: dpct -p=. --out-root=./out --cuda-include-path="%cuda-path/include"  --stop-on-parse-err --extra-arg="-xc"  --extra-arg="-I%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/c_feature_file_db/out/c_feature_file_db.c.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST %T/c_feature_file_db/out/c_feature_file_db.c.dp.cpp -o %T/c_feature_file_db/out/c_feature_file_db.c.dp.o %}
// RUN: cd ..
// RUN: rm -rf ./c_feature_file_db

#ifndef NO_BUILD_TEST
//CHECK:#include <sycl/sycl.hpp>
//CHECK:#include <dpct/dpct.hpp>
#include "cuda_runtime.h"
#include <stdio.h>

void func(int N, double re[][1<<N]) {
  printf("Hello from bindArraysToStackComplexMatrixN\n");
}

int main(int argc, char** argv) {
  const int N = 4;
  double a[1<<(N)][1<<(N)];

  func(N, a);
  return 0;
}
#endif
