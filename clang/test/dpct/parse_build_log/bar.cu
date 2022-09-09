// UNSUPPORTED: -windows-
// intercept-build only supports Linux 
// RUN: cp %S/build_log.txt %T/
// RUN: cd %T
// RUN: intercept-build -vvv  --parse-build-log ./build_log.txt   --in-root=./
// RUN: cat %S/compile_commands.json_ref  >%T/check_compilation_db.txt
// RUN: cat %T/compile_commands.json >>%T/check_compilation_db.txt
// RUN: FileCheck --match-full-lines --input-file %T/check_compilation_db.txt %T/check_compilation_db.txt

#include "header.cuh"
#include "cuda_runtime.h"
#include <stdio.h>

__global__ void bar(){}