// RUN: dpct --format-range=none -out-root %T/replace-align %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/replace-align/replace-align.dp.cpp

// CHECK:typedef struct __dpct_align__(4) dpct_type_{{[a-f0-9]+}}
typedef struct __align__(4)
{
    unsigned char r, g, b, a;
}
T0;

// CHECK:class __dpct_align__(8) T1 {
class __align__(8) T1 {
    unsigned int l, a;
};

// CHECK:struct __attribute__((aligned(16))) T2
struct __attribute__((aligned(16))) T2
{
    unsigned int r, g, b;
};


