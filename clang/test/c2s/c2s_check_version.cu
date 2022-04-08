// RUN: c2s --version > %T/c2s_check_version.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/c2s_check_version.txt

//CHECK: Intel(R) DPC++ Compatibility Tool version 2022.1.0. Codebase{{(.*)}}
__global__ void hello(){
}

