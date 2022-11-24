// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/type/counting_iterator %S/counting_iterator.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/type/counting_iterator/counting_iterator.dp.cpp --match-full-lines %s

// CHECK:#include <oneapi/dpl/iterator>
#include <cub/cub.cuh>
#include <stdio.h>

// CHECK:void test() {
// CHECK:oneapi::dpl::counting_iterator<int> iter(0);
// CHECK:for (int i = 0; i < 100; i += 2) {
// CHECK:printf("%d\n", iter[i]);
// CHECK:}
// CHECK:}
void test() {
    cub::CountingInputIterator<int> iter(0);
    for (int i = 0; i < 100; i += 2) {
        printf("%d\n", iter[i]);
    }
}
