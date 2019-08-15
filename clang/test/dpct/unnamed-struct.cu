// RUN: dpct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck --input-file %T/unnamed-struct.dp.cpp --match-full-lines %s

//CHECK: struct __sycl_align__(4) dpct_type_{{[a-f0-9]+}}
struct __align__(4)
{
    unsigned i;
} A;

//CHECK: typedef class dpct_type_{{[a-f0-9]+}}{
typedef class{
    unsigned i;
} T1;


//CHECK: typedef struct dpct_type_{{[a-f0-9]+}}
typedef struct
	: public T1
{
    unsigned j;
} T2;

//CHECK: class dpct_type_{{[a-f0-9]+}}: public T2 {
class: public T2 {
    unsigned k;
} B;
