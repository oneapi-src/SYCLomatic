// RUN: c2s -in-root %S -out-root %T %S/inc/a.cu %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file %T/inc/common.h --match-full-lines %S/inc/common.h
#include"inc/inc.h"
#include"inc/common.h"
// CHECK: int main(){
int main(){
}
