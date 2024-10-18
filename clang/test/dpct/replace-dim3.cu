// FIXME
// UNSUPPORTED: system-windows
// RUN: dpct --format-range=none --usm-level=none -out-root %T/replace-dim3 %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/replace-dim3/replace-dim3.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST  %T/replace-dim3/replace-dim3.dp.cpp -o %T/replace-dim3/replace-dim3.dp.o %}

#include <cstdio>
#include <algorithm>

#ifndef NO_BUILD_TEST
#define NUM 23
#define CALL_FUNC(func) func()

// CHECK: #define DIM3_DEFAULT_VAR(name) dpct::dim3 name
#define DIM3_DEFAULT_VAR(name) dim3 name

// CHECK: void func(dpct::dim3 a, dpct::dim3 b, dpct::dim3 c, dpct::dim3 d) {
void func(dim3 a, dim3 b, dim3 c, dim3 d) {
}

// CHECK: void test(const dpct::dim3& a, const dpct::dim3& b) {
void test(const dim3& a, const dim3& b) {
}

// CHECK: void test(dpct::dim3&& a, dpct::dim3&& b) {
void test(dim3&& a, dim3&& b) {
}

// CHECK: void test(const dpct::dim3* a, const dpct::dim3* b) {
void test(const dim3* a, const dim3* b) {
}

// CHECK: void test(const dpct::dim3** a, const dpct::dim3** b) {
void test(const dim3** a, const dim3** b) {
}

__global__ void kernel(int dim) {}

int main() {
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.out_of_order_queue();
  // range default constructor does the right thing.
  // CHECK: dpct::dim3 deflt;
  dim3 deflt;

  // CHECK: dpct::dim3 round1(1);
  dim3 round1(1);
  // CHECK: dpct::dim3 round1_1(NUM);
  dim3 round1_1(NUM);

  // CHECK: dpct::dim3 round2(2, 1);
  dim3 round2(2, 1);
  // CHECK: dpct::dim3 round2_1(NUM, NUM);
  dim3 round2_1(NUM, NUM);

  // CHECK: dpct::dim3 assign = 32;
  dim3 assign = 32;
  // CHECK: dpct::dim3 assign_1 = NUM;
  dim3 assign_1 = NUM;

  // CHECK: dpct::dim3 castini = (dpct::dim3)4;
  dim3 castini = (dim3)4;
  // CHECK: dpct::dim3 castini_1 = (dpct::dim3)NUM;
  dim3 castini_1 = (dim3)NUM;

  // CHECK: dpct::dim3 castini2 = dpct::dim3(2, 2);
  dim3 castini2 = dim3(2, 2);
  // CHECK: dpct::dim3 castini2_1 = dpct::dim3(NUM, NUM);
  dim3 castini2_1 = dim3(NUM, NUM);

  // CHECK: dpct::dim3 castini3 = dpct::dim3(3, 1, 10);
  dim3 castini3 = dim3(3, 1, 10);
  // CHECK: dpct::dim3 castini3_1 = dpct::dim3(NUM, NUM, NUM);
  dim3 castini3_1 = dim3(NUM, NUM, NUM);

  // CHECK: deflt = dpct::dim3(3);
  deflt = dim3(3);
  // CHECK: deflt = dpct::dim3(NUM);
  deflt = dim3(NUM);
  // CHECK: deflt = 5;
  deflt = 5;
  // CHECK: deflt = ((NUM%32 == 0) ? NUM/32 : (NUM/32 + 1));
  deflt = ((NUM%32 == 0) ? NUM/32 : (NUM/32 + 1));

  // CHECK: dpct::dim3 copyctor1 = dpct::dim3((dpct::dim3)33);
  dim3 copyctor1 = dim3((dim3)33);
  // CHECK: dpct::dim3 copyctor1_1 = dpct::dim3((dpct::dim3)NUM);
  dim3 copyctor1_1 = dim3((dim3)NUM);

  // CHECK: dpct::dim3 copyctor2 = dpct::dim3(copyctor1);
  dim3 copyctor2 = dim3(copyctor1);

  // CHECK: dpct::dim3 copyctor3(copyctor1);
  dim3 copyctor3(copyctor1);

  // CHECK: func((dpct::dim3)1, dpct::dim3(1), dpct::dim3(2, 1), dpct::dim3(3, 2, 1));
  func((dim3)1, dim3(1), dim3(2, 1), dim3(3, 2, 1));
  // CHECK: func((dpct::dim3)NUM, dpct::dim3(NUM), dpct::dim3(NUM, NUM), dpct::dim3(NUM, NUM, NUM));
  func((dim3)NUM, dim3(NUM), dim3(NUM, NUM), dim3(NUM, NUM, NUM));
  // CHECK: func(1, 2, 3, 4);
  func(1, 2, 3, 4);
  // CHECK: func(NUM, NUM, NUM, NUM);
  func(NUM, NUM, NUM, NUM);
  // CHECK: func(deflt, dpct::dim3(deflt), (dpct::dim3)deflt, 2 + 3 * 3);
  func(deflt, dim3(deflt), (dim3)deflt, 2 + 3 * 3);
  // CHECK: func(deflt, dpct::dim3(deflt), (dpct::dim3)deflt, NUM + NUM * NUM);
  func(deflt, dim3(deflt), (dim3)deflt, NUM + NUM * NUM);

  // CHECK: dpct::dim3 test(1, 2, 3);
  dim3 test(1, 2, 3);
  // CHECK: dpct::dim3 test_1(NUM, NUM, NUM);
  dim3 test_1(NUM, NUM, NUM);

  // CHECK: int b = test.x + test. y + test .z;
  int b = test.x + test. y + test .z;
  // CHECK: dpct::dim3 *p = &test;
  dim3 *p = &test;
  // CHECK: dpct::dim3 **pp = &p;
  dim3 **pp = &p;

  // CHECK: int a = p->x + p->y + p->z;
  int a = p->x + p->y + p->z;
  // CHECK: int aa = (*pp)->x + (*pp)->y + (*pp)->z;
  int aa = (*pp)->x + (*pp)->y + (*pp)->z;

  struct  container
  {
    unsigned int x, y, z;
    // CHECK: dpct::dim3 w;
    dim3 w;
    // CHECK: dpct::dim3 *pw;
    dim3 *pw;
    // CHECK: dpct::dim3 **ppw;
    dim3 **ppw;
  };
  typedef  struct container container;

  container t;

  // CHECK: int c = t.w.x + t.w.y + t.w.z;
  int c = t.w.x + t.w.y + t.w.z;
  // CHECK: int c2 = t.pw->x + t.pw->y + t.pw->z;
  int c2 = t.pw->x + t.pw->y + t.pw->z;
  // CHECK: int c3 = (*t.ppw)->x + (*t.ppw)->y + (*t.ppw)->z;
  int c3 = (*t.ppw)->x + (*t.ppw)->y + (*t.ppw)->z;

  // CHECK: dpct::dim3 d3_1(test.x);
  dim3 d3_1(test.x);
  // CHECK: dpct::dim3 d3_2(test.x + 1);
  dim3 d3_2(test.x + 1);
  // CHECK: dpct::dim3 d3_2_1(static_cast<unsigned>(test.x + 32));
  dim3 d3_2_1(static_cast<unsigned>(test.x + 32));
  // CHECK: dpct::dim3 d3_2_2(test.x + NUM);
  dim3 d3_2_2(test.x + NUM);
  // CHECK: dpct::dim3 d3_3(2 + test.x + 1);
  dim3 d3_3(2 + test.x + 1);
  // CHECK: dpct::dim3 d3_3_1(32 + test.x + 64);
  dim3 d3_3_1(32 + test.x + 64);
  // CHECK: dpct::dim3 d3_3_2(NUM + test.x + NUM);
  dim3 d3_3_2(NUM + test.x + NUM);
  // CHECK: dpct::dim3 d3_4(test.x, test.y);
  dim3 d3_4(test.x, test.y);
  // CHECK: dpct::dim3 d3_5(test.x, test.y, test.z);
  dim3 d3_5(test.x, test.y, test.z);
  // CHECK: dpct::dim3 d3_6 = dpct::dim3(test.x + 1, 2 + test.y, 3 + test.z + 4);
  dim3 d3_6 = dim3(test.x + 1, 2 + test.y, 3 + test.z + 4);
  // CHECK: dpct::dim3 d3_6_1 = dpct::dim3(test.x + 111, 112 + test.y, 113 + test.z + 114);
  dim3 d3_6_1 = dim3(test.x + 111, 112 + test.y, 113 + test.z + 114);
  // CHECK: dpct::dim3 d3_6_2 = dpct::dim3(test.x + NUM, NUM + test.y, NUM + test.z + NUM);
  dim3 d3_6_2 = dim3(test.x + NUM, NUM + test.y, NUM + test.z + NUM);
  // todoCHECK: dpct::dim3 d3_6_3 = dpct::dim3(ceil(test.x + NUM), NUM + test.y, NUM + test.z + NUM);
  dim3 d3_6_3 = dim3(ceil(test.x + NUM), NUM + test.y, NUM + test.z + NUM);
  // CHECK: dpct::dim3 gpu_blocks(1 / (d3_6_3.x * 200));
  dim3 gpu_blocks(1 / (d3_6_3.x * 200));
  // CHECK:   q_ct1.parallel_for<dpct_kernel_name<class kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           kernel(d3_6.x);
  // CHECK-NEXT:         });
  kernel<<<1, 1>>>(d3_6.x);
  // CHECK:   q_ct1.parallel_for<dpct_kernel_name<class kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, NUM) * sycl::range<3>(1, 1, NUM), sycl::range<3>(1, 1, NUM)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           kernel(d3_6.x);
  // CHECK-NEXT:         });
  kernel<<<NUM, NUM>>>(d3_6.x);
}

template<typename T>
__host__ __device__ T getgriddim(T totallen, T blockdim)
{
    return (totallen + blockdim - (T)1) / blockdim;
}

template<typename T>
static void memsetCuda(T * d_mem, T v, int n)
{
  // CHECK: dpct::dim3 dimBlock(256);
  // CHECK: dpct::dim3 dimGrid_2(std::max(2048, 3));
  // CHECK: dpct::dim3 dimGrid_1(std::max(2048, 3));
  // CHECK: std::min(2048, getgriddim<int>(n, dimBlock.x));
  // CHECK: dpct::dim3 dimGrid(std::min(2048, getgriddim<int>(n, dimBlock.x)));
  dim3 dimBlock(256);
  dim3 dimGrid_2(max(2048, 3));
  dim3 dimGrid_1(std::max(2048, 3));
  std::min(2048, getgriddim<int>(n, dimBlock.x));
  dim3 dimGrid(std::min(2048, getgriddim<int>(n, dimBlock.x)));
}

void test() {
    // TODO: Need to add test cases related to the situations below.
    // 1. if/while condition stmt
    // 2. macro stmt
    // 3. vec field address assignment expr, such as int i=&a.x
    // 4. one dimension vec, such as char1

    void *d_dst = NULL;
    FILE* dumpfile = NULL;

    // CHECK: sycl::uchar4* h_dst = (sycl::uchar4*) malloc(3*sizeof(sycl::uchar4));
    uchar4* h_dst = (uchar4*) malloc(3*sizeof(uchar4));

    for (int32_t i = 0; i < 3; ++i)
    {
        // CHECK: fwrite(&h_dst[i].x(), sizeof(char), 1, dumpfile);
        fwrite(&h_dst[i].x, sizeof(char), 1, dumpfile);

        // CHECK: fwrite(&h_dst[i].y(), sizeof(char), 1, dumpfile);
        fwrite(&h_dst[i].y, sizeof(char), 1, dumpfile);

        // CHECK: fwrite(&h_dst[i].z(), sizeof(char), 1, dumpfile);
        fwrite(&h_dst[i].z, sizeof(char), 1, dumpfile);
    }

    // CHECK: sycl::uchar4 data;
    uchar4 data;

    // CHECK: *(&data.x()) = 'a';
    *(&data.x) = 'a';
}

// CHECK:struct wrap {
// CHECK-NEXT:  sycl::float3 f3;
// CHECK-NEXT:};
struct wrap {
  float3 f3;
};


// CHECK: void kernel_foo(float *a, wrap *mt, unsigned int N,
// CHECK-NEXT: const sycl::nd_item<3> &item_ct1) {
// CHECK-NEXT:   const unsigned int i = item_ct1.get_group(2)*item_ct1.get_local_range(2)+item_ct1.get_local_id(2);
// CHECK-NEXT:   if (i<N) {
// CHECK-NEXT:     dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(&mt[i].f3.x(), a[i]);
// CHECK-NEXT:   }
// CHECK-NEXT: }
__global__ void kernel_foo(float *a, wrap *mt, unsigned int N) {
  const unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i<N) {
    atomicAdd(&mt[i].f3.x, a[i]);
  }
}

// CHECK: void dim3_foo() {
// CHECK-NEXT:     DIM3_DEFAULT_VAR(block0);
// CHECK-NEXT:     CALL_FUNC( []() {
// CHECK-NEXT:         dpct::dim3 block1;
// CHECK-NEXT:         dpct::dim3 block2{};
// CHECK-NEXT:         dpct::dim3 block3(2);
// CHECK-NEXT:         dpct::dim3 block4(2,3);
// CHECK-NEXT:         dpct::dim3 block5(2,3,4);
// CHECK-NEXT:         DIM3_DEFAULT_VAR(block6);
// CHECK-NEXT:       });
// CHECK-NEXT: }
void dim3_foo() {
    DIM3_DEFAULT_VAR(block0);
    CALL_FUNC( []() {
        dim3 block1;
        dim3 block2{};
        dim3 block3(2);
        dim3 block4(2,3);
        dim3 block5(2,3,4);
        DIM3_DEFAULT_VAR(block6);
      });
}
#endif

// CHECK: class Dim3Struct {
// CHECK-NEXT:   Dim3Struct() : x(dpct::dim3(1, 2)) {}
// CHECK-NEXT:   dpct::dim3 x = dpct::dim3(3, 4);
// CHECK-NEXT:   void f() { dpct::dim3(5, 6); }
// CHECK-NEXT: };
class Dim3Struct {
  Dim3Struct() : x(dim3(1, 2)) {}
  dim3 x = dim3(3, 4);
  void f() { dim3(5, 6); }
};

struct A {
  int x;
  dim3 y;
  int z;
};
struct B {
  int x;
  A y;
  dim3 z;
};

int dim3_implicit_ctor() {
  dim3 d;
  d.x = 5;
  // CHECK: B b1 = {};
  B b1 = {};
  // CHECK: B b2 = {0};
  B b2 = {0};
  // CHECK: B b3 = {0, {}};
  B b3 = {0, {}};
  // CHECK: B b4 = {0, {1}};
  B b4 = {0, {1}};
  // CHECK: B b5 = {0, {1, {1}}};
  B b5 = {0, {1, {1}}};
}
