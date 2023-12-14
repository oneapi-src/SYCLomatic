// UNSUPPORTED: system-linux
// RUN: cd %T
// RUN: mkdir c_feature_file_db_win
// RUN: cd c_feature_file_db_win
// RUN: cat %s > c_feature_file_db_win.c
// RUN: echo "[" > compile_commands.json
// RUN: echo "    {" >> compile_commands.json
// RUN: echo "        \"command\": \"CudaCompile c_feature_file_db_win.c\"," >> compile_commands.json
// RUN: echo "        \"directory\": \"%/T/c_feature_file_db_win\"," >> compile_commands.json
// RUN: echo "        \"file\": \"%/T/c_feature_file_db_win/c_feature_file_db_win.c\"" >> compile_commands.json
// RUN: echo "    }" >> compile_commands.json
// RUN: echo "]" >> compile_commands.json
// RUN: dpct -p=. --out-root=./out --cuda-include-path="%cuda-path/include" --extra-arg="-xc"  --stop-on-parse-err --extra-arg="-I%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/c_feature_file_db_win/out/c_feature_file_db_win.c.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/c_feature_file_db_win/out/c_feature_file_db_win.c.dp.cpp -o %T/c_feature_file_db_win/out/c_feature_file_db_win.c.dp.o %}
// RUN: cd ..
// RUN: rm -rf ./c_feature_file_db_win

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