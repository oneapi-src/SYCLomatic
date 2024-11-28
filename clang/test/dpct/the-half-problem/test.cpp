// RUN: cp -r %S/* %T/..
// RUN: dpct -p ./compile_commands.json --in-root=./src --out-root=%T/out --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %S/src/non-cuda-half.cu --match-full-lines --input-file %T/out/non-cuda-half.dp.cpp
// RUN: FileCheck %S/src/cuda-half.cu --match-full-lines --input-file %T/out/cuda-half.dp.cpp
