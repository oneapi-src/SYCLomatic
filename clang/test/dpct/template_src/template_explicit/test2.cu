// RUN: dpct --format-range=none -out-root %T/out %s %S/test.cu --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %S/test.hpp --match-full-lines --input-file %T/out/test.hpp
// RUN: %if build_lit %{icpx -c -fsycl %T/out/test.dp.cpp -o %T/out/test.dp.o %}
// RUN: %if build_lit %{icpx -c -fsycl %T/out/test2.dp.cpp -o %T/out/test2.dp.o %}
#define TEST
#include "test.hpp"
#undef TEST
