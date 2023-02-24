// RUN: dpct --format-range=none --usm-level=none -in-root %S -out-root %T/cuda_arch_test %S/test_2.cu -extra-arg="-I %S" --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %S/test_2.h --match-full-lines --input-file %T/cuda_arch_test/test_2.h
#include"test_2.h"