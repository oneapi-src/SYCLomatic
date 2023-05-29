// RUN: dpct --format-range=none -out-root %T/sizeof_vec3 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/sizeof_vec3/sizeof_vec3.dp.cpp --match-full-lines %s

#include <cuda_runtime.h>

/* char3 */

void test_char3_canonical_type() {
  char3 fp;

  // CHECK: DPCT1083:{{[0-9]+}}: The size of char3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  int size = sizeof(fp);

  // CHECK: DPCT1083:{{[0-9]+}}: The size of char3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  size = sizeof(char3);
  (void)size;
}

void test_char3_typedef() {
  typedef char3 fp3;
  fp3 a;

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka char3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  int size = sizeof(fp3);

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka char3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  size = sizeof(a);
  (void)size;
}

void test_char3_using() {
  using fp3 = char3;
  fp3 a;

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka char3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  int size = sizeof(fp3);

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka char3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  size = sizeof(a);
  (void)size;
}

// CHECK: void char3_shared_kernel(int *a, int *b, int *c) {
// CHECK-NOT: DPCT1083:{{[0-9]+}}: The size of char3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK: }
__global__ void char3_shared_kernel() {
  __shared__ int a[sizeof(char3) * 3], b[sizeof(char3)], c[10];
}

// CHECK: void char3_noshared_kernel() {
// CHECK: DPCT1083:{{[0-9]+}}: The size of char3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK: int a[sizeof(sycl::char3)];
// CHECK: }
__global__ void char3_noshared_kernel() {
  int a[sizeof(char3)];
}

// CHECK: void test_char3_shared() {
// CHECK: dpct::get_default_queue().submit(
// CHECK-NEXT: [&](sycl::handler &cgh) {
// CHECK: /*
// CHECK-NEXT: DPCT1083:{{[0-9]+}}: The size of local memory in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK-NEXT: */
// CHECK-NEXT: sycl::local_accessor<int, 1> a_acc_ct1(sycl::range<1>(sizeof(sycl::char3) * 3), cgh);
// CHECK: /*
// CHECK-NEXT: DPCT1083:{{[0-9]+}}: The size of local memory in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK-NEXT: */
// CHECK-NEXT: sycl::local_accessor<int, 1> b_acc_ct1(sycl::range<1>(sizeof(sycl::char3)), cgh);
// CHECK-NEXT: sycl::local_accessor<int, 1> c_acc_ct1(sycl::range<1>(10), cgh);
// CHECK: });
// CHECK: }
void test_char3_shared() {
  char3_shared_kernel<<<1, 1>>>();
}

/* uchar3 */ 

void test_uchar3_canonical_type() {
  uchar3 fp;

  // CHECK: DPCT1083:{{[0-9]+}}: The size of uchar3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  int size = sizeof(fp);

  // CHECK: DPCT1083:{{[0-9]+}}: The size of uchar3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  size = sizeof(uchar3);
  (void)size;
}

void test_uchar3_typedef() {
  typedef uchar3 fp3;
  fp3 a;

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka uchar3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  int size = sizeof(fp3);

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka uchar3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  size = sizeof(a);
  (void)size;
}

void test_uchar3_using() {
  using fp3 = uchar3;
  fp3 a;

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka uchar3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  int size = sizeof(fp3);

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka uchar3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  size = sizeof(a);
  (void)size;
}

// CHECK: void uchar3_shared_kernel(int *a, int *b, int *c) {
// CHECK-NOT: DPCT1083:{{[0-9]+}}: The size of uchar3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK: }
__global__ void uchar3_shared_kernel() {
  __shared__ int a[sizeof(uchar3) * 3], b[sizeof(uchar3)], c[10];
}

// CHECK: void uchar3_noshared_kernel() {
// CHECK: DPCT1083:{{[0-9]+}}: The size of uchar3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK: int a[sizeof(sycl::uchar3)];
// CHECK: }
__global__ void uchar3_noshared_kernel() {
  int a[sizeof(uchar3)];
}

// CHECK: void test_uchar3_shared() {
// CHECK: dpct::get_default_queue().submit(
// CHECK-NEXT: [&](sycl::handler &cgh) {
// CHECK: /*
// CHECK-NEXT: DPCT1083:{{[0-9]+}}: The size of local memory in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK-NEXT: */
// CHECK-NEXT: sycl::local_accessor<int, 1> a_acc_ct1(sycl::range<1>(sizeof(sycl::uchar3) * 3), cgh);
// CHECK: /*
// CHECK-NEXT: DPCT1083:{{[0-9]+}}: The size of local memory in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK-NEXT: */
// CHECK-NEXT: sycl::local_accessor<int, 1> b_acc_ct1(sycl::range<1>(sizeof(sycl::uchar3)), cgh);
// CHECK-NEXT: sycl::local_accessor<int, 1> c_acc_ct1(sycl::range<1>(10), cgh);
// CHECK: });
// CHECK: }
void test_uchar3_shared() {
  uchar3_shared_kernel<<<1, 1>>>();
}

/* short3 */

void test_short3_canonical_type() {
  short3 fp;

  // CHECK: DPCT1083:{{[0-9]+}}: The size of short3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  int size = sizeof(fp);

  // CHECK: DPCT1083:{{[0-9]+}}: The size of short3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  size = sizeof(short3);
  (void)size;
}

void test_short3_typedef() {
  typedef short3 fp3;
  fp3 a;

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka short3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  int size = sizeof(fp3);

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka short3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  size = sizeof(a);
  (void)size;
}

void test_short3_using() {
  using fp3 = short3;
  fp3 a;

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka short3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  int size = sizeof(fp3);

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka short3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  size = sizeof(a);
  (void)size;
}

// CHECK: void short3_shared_kernel(int *a, int *b, int *c) {
// CHECK-NOT: DPCT1083:{{[0-9]+}}: The size of short3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK: }
__global__ void short3_shared_kernel() {
  __shared__ int a[sizeof(short3) * 3], b[sizeof(short3)], c[10];
}

// CHECK: void short3_noshared_kernel() {
// CHECK: DPCT1083:{{[0-9]+}}: The size of short3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK: int a[sizeof(sycl::short3)];
// CHECK: }
__global__ void short3_noshared_kernel() {
  int a[sizeof(short3)];
}

// CHECK: void test_short3_shared() {
// CHECK: dpct::get_default_queue().submit(
// CHECK-NEXT: [&](sycl::handler &cgh) {
// CHECK: /*
// CHECK-NEXT: DPCT1083:{{[0-9]+}}: The size of local memory in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK-NEXT: */
// CHECK-NEXT: sycl::local_accessor<int, 1> a_acc_ct1(sycl::range<1>(sizeof(sycl::short3) * 3), cgh);
// CHECK: /*
// CHECK-NEXT: DPCT1083:{{[0-9]+}}: The size of local memory in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK-NEXT: */
// CHECK-NEXT: sycl::local_accessor<int, 1> b_acc_ct1(sycl::range<1>(sizeof(sycl::short3)), cgh);
// CHECK-NEXT: sycl::local_accessor<int, 1> c_acc_ct1(sycl::range<1>(10), cgh);
// CHECK: });
// CHECK: }
void test_short3_shared() {
  short3_shared_kernel<<<1, 1>>>();
}

/* ushort3 */

void test_ushort3_canonical_type() {
  ushort3 fp;

  // CHECK: DPCT1083:{{[0-9]+}}: The size of ushort3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  int size = sizeof(fp);

  // CHECK: DPCT1083:{{[0-9]+}}: The size of ushort3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  size = sizeof(ushort3);
  (void)size;
}

void test_ushort3_typedef() {
  typedef ushort3 fp3;
  fp3 a;

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka ushort3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  int size = sizeof(fp3);

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka ushort3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  size = sizeof(a);
  (void)size;
}

void test_ushort3_using() {
  using fp3 = ushort3;
  fp3 a;

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka ushort3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  int size = sizeof(fp3);

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka ushort3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  size = sizeof(a);
  (void)size;
}

// CHECK: void ushort3_shared_kernel(int *a, int *b, int *c) {
// CHECK-NOT: DPCT1083:{{[0-9]+}}: The size of ushort3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK: }
__global__ void ushort3_shared_kernel() {
  __shared__ int a[sizeof(ushort3) * 3], b[sizeof(ushort3)], c[10];
}

// CHECK: void ushort3_noshared_kernel() {
// CHECK: DPCT1083:{{[0-9]+}}: The size of ushort3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK: int a[sizeof(sycl::ushort3)];
// CHECK: }
__global__ void ushort3_noshared_kernel() {
  int a[sizeof(ushort3)];
}

// CHECK: void test_ushort3_shared() {
// CHECK: dpct::get_default_queue().submit(
// CHECK-NEXT: [&](sycl::handler &cgh) {
// CHECK: /*
// CHECK-NEXT: DPCT1083:{{[0-9]+}}: The size of local memory in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK-NEXT: */
// CHECK-NEXT: sycl::local_accessor<int, 1> a_acc_ct1(sycl::range<1>(sizeof(sycl::ushort3) * 3), cgh);
// CHECK: /*
// CHECK-NEXT: DPCT1083:{{[0-9]+}}: The size of local memory in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK-NEXT: */
// CHECK-NEXT: sycl::local_accessor<int, 1> b_acc_ct1(sycl::range<1>(sizeof(sycl::ushort3)), cgh);
// CHECK-NEXT: sycl::local_accessor<int, 1> c_acc_ct1(sycl::range<1>(10), cgh);
// CHECK: });
// CHECK: }
void test_ushort3_shared() {
  ushort3_shared_kernel<<<1, 1>>>();
}

/* int3 */

void test_int3_canonical_type() {
  int3 fp;

  // CHECK: DPCT1083:{{[0-9]+}}: The size of int3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  int size = sizeof(fp);

  // CHECK: DPCT1083:{{[0-9]+}}: The size of int3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  size = sizeof(int3);
  (void)size;
}

void test_int3_typedef() {
  typedef int3 fp3;
  fp3 a;

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka int3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  int size = sizeof(fp3);

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka int3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  size = sizeof(a);
  (void)size;
}

void test_int3_using() {
  using fp3 = int3;
  fp3 a;

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka int3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  int size = sizeof(fp3);

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka int3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  size = sizeof(a);
  (void)size;
}

// CHECK: void int3_shared_kernel(int *a, int *b, int *c) {
// CHECK-NOT: DPCT1083:{{[0-9]+}}: The size of int3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK: }
__global__ void int3_shared_kernel() {
  __shared__ int a[sizeof(int3) * 3], b[sizeof(int3)], c[10];
}

// CHECK: void int3_noshared_kernel() {
// CHECK: DPCT1083:{{[0-9]+}}: The size of int3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK: int a[sizeof(sycl::int3)];
// CHECK: }
__global__ void int3_noshared_kernel() {
  int a[sizeof(int3)];
}

// CHECK: void test_int3_shared() {
// CHECK: dpct::get_default_queue().submit(
// CHECK-NEXT: [&](sycl::handler &cgh) {
// CHECK: /*
// CHECK-NEXT: DPCT1083:{{[0-9]+}}: The size of local memory in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK-NEXT: */
// CHECK-NEXT: sycl::local_accessor<int, 1> a_acc_ct1(sycl::range<1>(sizeof(sycl::int3) * 3), cgh);
// CHECK: /*
// CHECK-NEXT: DPCT1083:{{[0-9]+}}: The size of local memory in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK-NEXT: */
// CHECK-NEXT: sycl::local_accessor<int, 1> b_acc_ct1(sycl::range<1>(sizeof(sycl::int3)), cgh);
// CHECK-NEXT: sycl::local_accessor<int, 1> c_acc_ct1(sycl::range<1>(10), cgh);
// CHECK: });
// CHECK: }
void test_int3_shared() {
  int3_shared_kernel<<<1, 1>>>();
}

/* uint3 */

void test_uint3_canonical_type() {
  uint3 fp;

  // CHECK: DPCT1083:{{[0-9]+}}: The size of uint3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  int size = sizeof(fp);

  // CHECK: DPCT1083:{{[0-9]+}}: The size of uint3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  size = sizeof(uint3);
  (void)size;
}

void test_uint3_typedef() {
  typedef uint3 fp3;
  fp3 a;

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka uint3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  int size = sizeof(fp3);

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka uint3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  size = sizeof(a);
  (void)size;
}

void test_uint3_using() {
  using fp3 = uint3;
  fp3 a;

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka uint3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  int size = sizeof(fp3);

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka uint3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  size = sizeof(a);
  (void)size;
}

// CHECK: void uint3_shared_kernel(int *a, int *b, int *c) {
// CHECK-NOT: DPCT1083:{{[0-9]+}}: The size of uint3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK: }
__global__ void uint3_shared_kernel() {
  __shared__ int a[sizeof(uint3) * 3], b[sizeof(uint3)], c[10];
}

// CHECK: void uint3_noshared_kernel() {
// CHECK: DPCT1083:{{[0-9]+}}: The size of uint3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK: int a[sizeof(sycl::uint3)];
// CHECK: }
__global__ void uint3_noshared_kernel() {
  int a[sizeof(uint3)];
}

// CHECK: void test_uint3_shared() {
// CHECK: dpct::get_default_queue().submit(
// CHECK-NEXT: [&](sycl::handler &cgh) {
// CHECK: /*
// CHECK-NEXT: DPCT1083:{{[0-9]+}}: The size of local memory in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK-NEXT: */
// CHECK-NEXT: sycl::local_accessor<int, 1> a_acc_ct1(sycl::range<1>(sizeof(sycl::uint3) * 3), cgh);
// CHECK: /*
// CHECK-NEXT: DPCT1083:{{[0-9]+}}: The size of local memory in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK-NEXT: */
// CHECK-NEXT: sycl::local_accessor<int, 1> b_acc_ct1(sycl::range<1>(sizeof(sycl::uint3)), cgh);
// CHECK-NEXT: sycl::local_accessor<int, 1> c_acc_ct1(sycl::range<1>(10), cgh);
// CHECK: });
// CHECK: }
void test_uint3_shared() {
  uint3_shared_kernel<<<1, 1>>>();
}

/* long3 */

void test_long3_canonical_type() {
  long3 fp;

  // CHECK: DPCT1083:{{[0-9]+}}: The size of long3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  int size = sizeof(fp);

  // CHECK: DPCT1083:{{[0-9]+}}: The size of long3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  size = sizeof(long3);
  (void)size;
}

void test_long3_typedef() {
  typedef long3 fp3;
  fp3 a;

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka long3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  int size = sizeof(fp3);

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka long3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  size = sizeof(a);
  (void)size;
}

void test_long3_using() {
  using fp3 = long3;
  fp3 a;

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka long3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  int size = sizeof(fp3);

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka long3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  size = sizeof(a);
  (void)size;
}

// CHECK: void long3_shared_kernel(int *a, int *b, int *c) {
// CHECK-NOT: DPCT1083:{{[0-9]+}}:The size of long3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK: }
__global__ void long3_shared_kernel() {
  __shared__ int a[sizeof(long3) * 3], b[sizeof(long3)], c[10];
}

// CHECK: void long3_noshared_kernel() {
// CHECK: DPCT1083:{{[0-9]+}}: The size of long3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK: int a[sizeof(sycl::long3)];
// CHECK: }
__global__ void long3_noshared_kernel() {
  int a[sizeof(long3)];
}

// CHECK: void test_long3_shared() {
// CHECK: dpct::get_default_queue().submit(
// CHECK-NEXT: [&](sycl::handler &cgh) {
// CHECK: /*
// CHECK-NEXT: DPCT1083:{{[0-9]+}}: The size of local memory in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK-NEXT: */
// CHECK-NEXT: sycl::local_accessor<int, 1> a_acc_ct1(sycl::range<1>(sizeof(sycl::long3) * 3), cgh);
// CHECK: /*
// CHECK-NEXT: DPCT1083:{{[0-9]+}}: The size of local memory in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK-NEXT: */
// CHECK-NEXT: sycl::local_accessor<int, 1> b_acc_ct1(sycl::range<1>(sizeof(sycl::long3)), cgh);
// CHECK-NEXT: sycl::local_accessor<int, 1> c_acc_ct1(sycl::range<1>(10), cgh);
// CHECK: });
// CHECK: }
void test_long3_shared() {
  long3_shared_kernel<<<1, 1>>>();
}

/* ulong3 */

void test_ulong3_canonical_type() {
  ulong3 fp;

  // CHECK: DPCT1083:{{[0-9]+}}: The size of ulong3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  int size = sizeof(fp);

  // CHECK: DPCT1083:{{[0-9]+}}: The size of ulong3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  size = sizeof(ulong3);
  (void)size;
}

void test_ulong3_typedef() {
  typedef ulong3 fp3;
  fp3 a;

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka ulong3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  int size = sizeof(fp3);

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka ulong3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  size = sizeof(a);
  (void)size;
}

void test_ulong3_using() {
  using fp3 = ulong3;
  fp3 a;

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka ulong3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  int size = sizeof(fp3);

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka ulong3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  size = sizeof(a);
  (void)size;
}

// CHECK: void ulong3_shared_kernel(int *a, int *b, int *c) {
// CHECK-NOT: DPCT1083:{{[0-9]+}}: The size of ulong3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK: }
__global__ void ulong3_shared_kernel() {
  __shared__ int a[sizeof(ulong3) * 3], b[sizeof(ulong3)], c[10];
}

// CHECK: void ulong3_noshared_kernel() {
// CHECK: DPCT1083:{{[0-9]+}}: The size of ulong3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK: int a[sizeof(sycl::ulong3)];
// CHECK: }
__global__ void ulong3_noshared_kernel() {
  int a[sizeof(ulong3)];
}

// CHECK: void test_ulong3_shared() {
// CHECK: dpct::get_default_queue().submit(
// CHECK-NEXT: [&](sycl::handler &cgh) {
// CHECK: /*
// CHECK-NEXT: DPCT1083:{{[0-9]+}}: The size of local memory in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK-NEXT: */
// CHECK-NEXT: sycl::local_accessor<int, 1> a_acc_ct1(sycl::range<1>(sizeof(sycl::ulong3) * 3), cgh);
// CHECK: /*
// CHECK-NEXT: DPCT1083:{{[0-9]+}}: The size of local memory in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK-NEXT: */
// CHECK-NEXT: sycl::local_accessor<int, 1> b_acc_ct1(sycl::range<1>(sizeof(sycl::ulong3)), cgh);
// CHECK-NEXT: sycl::local_accessor<int, 1> c_acc_ct1(sycl::range<1>(10), cgh);
// CHECK: });
// CHECK: }
void test_ulong3_shared() {
  ulong3_shared_kernel<<<1, 1>>>();
}

/* longlong3 */

void test_longlong3_canonical_type() {
  longlong3 fp;

  // CHECK: DPCT1083:{{[0-9]+}}: The size of longlong3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  int size = sizeof(fp);

  // CHECK: DPCT1083:{{[0-9]+}}: The size of longlong3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  size = sizeof(longlong3);
  (void)size;
}

void test_longlong3_typedef() {
  typedef longlong3 fp3;
  fp3 a;

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka longlong3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  int size = sizeof(fp3);

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka longlong3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  size = sizeof(a);
  (void)size;
}

void test_longlong3_using() {
  using fp3 = longlong3;
  fp3 a;

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka longlong3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  int size = sizeof(fp3);

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka longlong3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  size = sizeof(a);
  (void)size;
}

// CHECK: void longlong3_shared_kernel(int *a, int *b, int *c) {
// CHECK-NOT: DPCT1083:{{[0-9]+}}: The size of longlong3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK: }
__global__ void longlong3_shared_kernel() {
  __shared__ int a[sizeof(longlong3) * 3], b[sizeof(longlong3)], c[10];
}

// CHECK: void longlong3_noshared_kernel() {
// CHECK: DPCT1083:{{[0-9]+}}: The size of longlong3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK: int a[sizeof(sycl::vec<std::int64_t, 3>)];
// CHECK: }
__global__ void longlong3_noshared_kernel() {
  int a[sizeof(longlong3)];
}

// CHECK: void test_longlong3_shared() {
// CHECK: dpct::get_default_queue().submit(
// CHECK-NEXT: [&](sycl::handler &cgh) {
// CHECK: /*
// CHECK-NEXT: DPCT1083:{{[0-9]+}}: The size of local memory in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK-NEXT: */
// CHECK-NEXT: sycl::local_accessor<int, 1> a_acc_ct1(sycl::range<1>(sizeof(sycl::vec<std::int64_t, 3>) * 3), cgh);
// CHECK: /*
// CHECK-NEXT: DPCT1083:{{[0-9]+}}: The size of local memory in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK-NEXT: */
// CHECK-NEXT: sycl::local_accessor<int, 1> b_acc_ct1(sycl::range<1>(sizeof(sycl::vec<std::int64_t, 3>)), cgh);
// CHECK-NEXT: sycl::local_accessor<int, 1> c_acc_ct1(sycl::range<1>(10), cgh);
// CHECK: });
// CHECK: }
void test_longlong3_shared() {
  longlong3_shared_kernel<<<1, 1>>>();
}

/* ulonglong3 */

void test_ulonglong3_canonical_type() {
  ulonglong3 fp;

  // CHECK: DPCT1083:{{[0-9]+}}: The size of ulonglong3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  int size = sizeof(fp);

  // CHECK: DPCT1083:{{[0-9]+}}: The size of ulonglong3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  size = sizeof(ulonglong3);
  (void)size;
}

void test_ulonglong3_typedef() {
  typedef ulonglong3 fp3;
  fp3 a;

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka ulonglong3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  int size = sizeof(fp3);

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka ulonglong3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  size = sizeof(a);
  (void)size;
}

void test_ulonglong3_using() {
  using fp3 = ulonglong3;
  fp3 a;

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka ulonglong3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  int size = sizeof(fp3);

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka ulonglong3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  size = sizeof(a);
  (void)size;
}

// CHECK: void ulonglong3_shared_kernel(int *a, int *b, int *c) {
// CHECK-NOT: DPCT1083:{{[0-9]+}}: The size of ulonglong3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK: }
__global__ void ulonglong3_shared_kernel() {
  __shared__ int a[sizeof(ulonglong3) * 3], b[sizeof(ulonglong3)], c[10];
}

// CHECK: void ulonglong3_noshared_kernel() {
// CHECK: DPCT1083:{{[0-9]+}}:  The size of ulonglong3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK: int a[sizeof(sycl::vec<std::uint64_t, 3>)];
// CHECK: }
__global__ void ulonglong3_noshared_kernel() {
  int a[sizeof(ulonglong3)];
}

// CHECK: void test_ulonglong3_shared() {
// CHECK: dpct::get_default_queue().submit(
// CHECK-NEXT: [&](sycl::handler &cgh) {
// CHECK: /*
// CHECK-NEXT: DPCT1083:{{[0-9]+}}: The size of local memory in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK-NEXT: */
// CHECK-NEXT: sycl::local_accessor<int, 1> a_acc_ct1(sycl::range<1>(sizeof(sycl::vec<std::uint64_t, 3>) * 3), cgh);
// CHECK: /*
// CHECK-NEXT: DPCT1083:{{[0-9]+}}: The size of local memory in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK-NEXT: */
// CHECK-NEXT: sycl::local_accessor<int, 1> b_acc_ct1(sycl::range<1>(sizeof(sycl::vec<std::uint64_t, 3>)), cgh);
// CHECK-NEXT: sycl::local_accessor<int, 1> c_acc_ct1(sycl::range<1>(10), cgh);
// CHECK: });
// CHECK: }
void test_ulonglong3_shared() {
  ulonglong3_shared_kernel<<<1, 1>>>();
}


/* float3 */

void test_float3_canonical_type() {
  float3 fp;

  // CHECK: DPCT1083:{{[0-9]+}}: The size of float3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  int size = sizeof(fp);

  // CHECK: DPCT1083:{{[0-9]+}}: The size of float3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  size = sizeof(float3);
  (void)size;
}

void test_float3_typedef() {
  typedef float3 fp3;
  fp3 a;

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka float3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  int size = sizeof(fp3);

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka float3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  size = sizeof(a);
  (void)size;
}

void test_float3_using() {
  using fp3 = float3;
  fp3 a;

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka float3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  int size = sizeof(fp3);

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka float3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  size = sizeof(a);
  (void)size;
}

// CHECK: void float3_shared_kernel(int *a, int *b, int *c) {
// CHECK-NOT: DPCT1083:{{[0-9]+}}: The size of float3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK: }
__global__ void float3_shared_kernel() {
  __shared__ int a[sizeof(float3) * 3], b[sizeof(float3)], c[10];
}

// CHECK: void float3_noshared_kernel() {
// CHECK: DPCT1083:{{[0-9]+}}: The size of float3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK: int a[sizeof(sycl::float3)];
// CHECK: }
__global__ void float3_noshared_kernel() {
  int a[sizeof(float3)];
}

// CHECK: void test_float3_shared() {
// CHECK: dpct::get_default_queue().submit(
// CHECK-NEXT: [&](sycl::handler &cgh) {
// CHECK: /*
// CHECK-NEXT: DPCT1083:{{[0-9]+}}: The size of local memory in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK-NEXT: */
// CHECK-NEXT: sycl::local_accessor<int, 1> a_acc_ct1(sycl::range<1>(sizeof(sycl::float3) * 3), cgh);
// CHECK: /*
// CHECK-NEXT: DPCT1083:{{[0-9]+}}: The size of local memory in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK-NEXT: */
// CHECK-NEXT: sycl::local_accessor<int, 1> b_acc_ct1(sycl::range<1>(sizeof(sycl::float3)), cgh);
// CHECK-NEXT: sycl::local_accessor<int, 1> c_acc_ct1(sycl::range<1>(10), cgh);
// CHECK: });
// CHECK: }
void test_float3_shared() {
  float3_shared_kernel<<<1, 1>>>();
}

/* double3 */

void test_double3_canonical_type() {
  double3 fp;

  // CHECK: DPCT1083:{{[0-9]+}}: The size of double3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  int size = sizeof(fp);

  // CHECK: DPCT1083:{{[0-9]+}}: The size of double3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  size = sizeof(double3);
  (void)size;
}

void test_double3_typedef() {
  typedef double3 fp3;
  fp3 a;

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka double3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  int size = sizeof(fp3);

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka double3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  size = sizeof(a);
  (void)size;
}

void test_double3_using() {
  using fp3 = double3;
  fp3 a;

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka double3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  int size = sizeof(fp3);

  // CHECK: DPCT1083:{{[0-9]+}}: The size of fp3 (aka double3) in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
  size = sizeof(a);
  (void)size;
}

// CHECK: void double3_shared_kernel(int *a, int *b, int *c) {
// CHECK-NOT: DPCT1083:{{[0-9]+}}: The size of double3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK: }
__global__ void double3_shared_kernel() {
  __shared__ int a[sizeof(double3) * 3], b[sizeof(double3)], c[10];
}

// CHECK: void double3_noshared_kernel() {
// CHECK: DPCT1083:{{[0-9]+}}: The size of double3 in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK: int a[sizeof(sycl::double3)];
// CHECK: }
__global__ void double3_noshared_kernel() {
  int a[sizeof(double3)];
}

// CHECK: void test_double3_shared() {
// CHECK: dpct::get_default_queue().submit(
// CHECK-NEXT: [&](sycl::handler &cgh) {
// CHECK: /*
// CHECK-NEXT: DPCT1083:{{[0-9]+}}: The size of local memory in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK-NEXT: */
// CHECK-NEXT: sycl::local_accessor<int, 1> a_acc_ct1(sycl::range<1>(sizeof(sycl::double3) * 3), cgh);
// CHECK: /*
// CHECK-NEXT: DPCT1083:{{[0-9]+}}: The size of local memory in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK-NEXT: */
// CHECK-NEXT: sycl::local_accessor<int, 1> b_acc_ct1(sycl::range<1>(sizeof(sycl::double3)), cgh);
// CHECK-NEXT: sycl::local_accessor<int, 1> c_acc_ct1(sycl::range<1>(10), cgh);
// CHECK: });
// CHECK: }
void test_double3_shared() {
  double3_shared_kernel<<<1, 1>>>();
}
