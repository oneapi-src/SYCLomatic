// RUN: dpct --format-range=none -out-root %T/array_size_fold %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/array_size_fold/array_size_fold.dp.cpp

#define S 3
class C {};
__global__ void f() {
  const int x = 2;
  // fold size expr:
  // CHECK: /*
  // CHECK-NEXT: DPCT1101:{{[0-9]+}}: 'S' expression was replaced with a value. Modify the code to use original expression, provided in comments, if it is correct.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::local_accessor<int, 1> fold1_acc_ct1(sycl::range<1>(3/*S*/), cgh);
  __shared__ int fold1[S];
  // CHECK: /*
  // CHECK-NEXT: DPCT1101:{{[0-9]+}}: 'x' expression was replaced with a value. Modify the code to use original expression, provided in comments, if it is correct.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::local_accessor<int, 1> fold2_acc_ct1(sycl::range<1>(2/*x*/), cgh);
  __shared__ int fold2[x];
  // CHECK: /*
  // CHECK-NEXT: DPCT1101:{{[0-9]+}}: 'sizeof(C) * 3' expression was replaced with a value. Modify the code to use original expression, provided in comments, if it is correct.
  // CHECK-NEXT: */
  // CHECK: /*
  // CHECK-NEXT: DPCT1083:{{[0-9]+}}: The size of local memory in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::local_accessor<int, 1> fold3_acc_ct1(sycl::range<1>(3/*sizeof(C) * 3*/), cgh);
  __shared__ int fold3[sizeof(C) * 3];
  // CHECK: /*
  // CHECK-NEXT: DPCT1101:{{[0-9]+}}: 'sizeof(x) * 3' expression was replaced with a value. Modify the code to use original expression, provided in comments, if it is correct.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::local_accessor<int, 1> fold4_acc_ct1(sycl::range<1>(12/*sizeof(x) * 3*/), cgh);
  __shared__ int fold4[sizeof(x) * 3];
  // CHECK: /*
  // CHECK-NEXT: DPCT1101:{{[0-9]+}}: 'sizeof(x * 3) * 3' expression was replaced with a value. Modify the code to use original expression, provided in comments, if it is correct.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::local_accessor<int, 1> fold5_acc_ct1(sycl::range<1>(12/*sizeof(x * 3) * 3*/), cgh);
  __shared__ int fold5[sizeof(x * 3) * 3];
  // CHECK: /*
  // CHECK-NEXT: DPCT1101:{{[0-9]+}}: 'S' expression was replaced with a value. Modify the code to use original expression, provided in comments, if it is correct.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1101:{{[0-9]+}}: 'S+1+S' expression was replaced with a value. Modify the code to use original expression, provided in comments, if it is correct.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::local_accessor<int, 2> fold6_acc_ct1(sycl::range<2>(3/*S*/, 7/*S+1+S*/), cgh);
  __shared__ int fold6[S][S+1+S];
  // not fold size expr:
  // CHECK: sycl::local_accessor<int, 1> unfold1_acc_ct1(sycl::range<1>(1 + 1), cgh);
  __shared__ int unfold1[1 + 1];
  // CHECK: sycl::local_accessor<int, 1> unfold2_acc_ct1(sycl::range<1>(sizeof(sycl::float3) * 3), cgh);
  __shared__ int unfold2[sizeof(float3) * 3];
}
int main() {
  f<<<1, 1>>>();
  return 0;
}
