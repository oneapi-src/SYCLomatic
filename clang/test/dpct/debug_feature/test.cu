// RUN: dpct --format-range=none  -out-root %T/debug_feature/test %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/../output_int2.json --match-full-lines %S/output_int2.json

#include <cuda.h>

void faketest(int2 a){
return;
}

int main(){
  int2 a;
  faketest(a);
}
