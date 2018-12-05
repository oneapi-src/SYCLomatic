// RUN: syclct -out-root %T %s -- -std=c++11  -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck %s --match-full-lines --input-file %T/replace-align.sycl.cpp

// CHECK:typedef struct __sycl_align__(4) SYCL_TYPE_{{[a-f0-9]+}}
typedef struct __align__(4)
{
    unsigned char r, g, b, a;
}
T0;

// CHECK:class __sycl_align__(8) T1 {
class __align__(8) T1 {
    unsigned int l, a;
};

// CHECK:struct __attribute__((aligned(16))) T2
struct __attribute__((aligned(16))) T2
{
    unsigned int r, g, b;
};

