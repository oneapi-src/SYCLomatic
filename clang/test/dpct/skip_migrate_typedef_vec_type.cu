// RUN: dpct --format-range=none -in-root %S -out-root %T/skip_migrate_typedef_vec_type %s -cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/skip_migrate_typedef_vec_type/skip_migrate_typedef_vec_type.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/skip_migrate_typedef_vec_type/skip_migrate_typedef_vec_type.dp.cpp -o %T/skip_migrate_typedef_vec_type/skip_migrate_typedef_vec_type.dp.o %}


#include <cuda_fp16.h>

namespace mma {
// CHECK: using half2 = sycl::half;
using half2 = __nv_half;
};
template <class T>
void test_t(T t) {
}
int main() {
// CHECK: mma::half2 test;
    mma::half2 test;
// CHECK: test_t<mma::half2>(test);
    test_t<mma::half2>(test);
    return 0;
}