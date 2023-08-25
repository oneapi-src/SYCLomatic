// Disable this case but not remove it to avoid pull down conflict
// UNSUPPORTED: system-linux
// UNSUPPORTED: system-windows
// RUN: dpct --version > %T/dpct_check_version.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/dpct_check_version.txt

//CHECK: dpct version 18.0.0. Codebase{{(.*)}}
__global__ void hello(){
}

