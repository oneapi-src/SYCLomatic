// RUN: dpct --format-range=none -out-root %T/ %s --cuda-include-path="%cuda-path/include" --extra-arg="-forward-unknown-to-host-compiler" > log.txt 2>&1
// RUN: FileCheck --input-file %T/parse_error_msgs.dp.cpp --match-full-lines %s
// RUN: grep -srn "dpct/c2s ignoring argument: .-forward-unknown-to-host-compile." ./log.txt

#include "cuda.h"

// CHECK: void g() {}
__global__ void g() {}

