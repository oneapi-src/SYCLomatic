// RUN: dpct --version > %T/dpct_check_version.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/dpct_check_version.txt

//CHECK: dpct version 1.0.0. Codebase{{(.*)}}
__global__ void hello(){
}

