// RUN: dpct --format-range=none --out-root %T %s %S/b.cu --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %S/k.inc --match-full-lines --input-file %T/k.inc

#include "k.inc"

template void foo38<double>();
template void foo38<float>();
