// RUN: dpct --format-range=none -out-root %Tt %s %S/test.cu --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %S/test.hpp --match-full-lines --input-file %Tt/test.hpp
// RUN: %if build_lit %{icpx -c -fsycl %Tt/test.dp.cpp -o %Tt/test.dp.o %}
// RUN: %if build_lit %{icpx -c -fsycl %Tt/test2.dp.cpp -o %Tt/test2.dp.o %}
#define TEST
#include "test.hpp"

#undef TEST

void test() {
}
