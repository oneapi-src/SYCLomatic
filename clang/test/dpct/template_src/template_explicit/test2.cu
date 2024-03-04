// RUN: dpct --format-range=none -out-root %T/template_src/template_explicit/out %s %S/test.cu --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %S/test.hpp --match-full-lines --input-file %T/template_src/template_explicit/out/test.hpp

#define TEST
#include "test.hpp"

#undef TEST

void test() {
}
