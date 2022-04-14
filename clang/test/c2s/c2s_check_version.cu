// RUN: c2s --version > %T/c2s_check_version.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/c2s_check_version.txt

//CHECK: c2s version 1.0.0. Codebase{{(.*)}}
__global__ void hello(){
}

