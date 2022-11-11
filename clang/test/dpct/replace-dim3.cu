// FIXME
// UNSUPPORTED: -windows-
// RUN: dpct --format-range=none --usm-level=none -out-root %T/replace-dim3 %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/replace-dim3/replace-dim3.dp.cpp --match-full-lines %s

#include <cstdio>
#include <algorithm>

#define NUM 23
#define CALL_FUNC(func) func()

// CHECK: #define DIM3_DEFAULT_VAR(name) sycl::range<3> name
#define DIM3_DEFAULT_VAR(name) dim3 name

// CHECK: void func(sycl::range<3> a, sycl::range<3> b, sycl::range<3> c, sycl::range<3> d) {
void func(dim3 a, dim3 b, dim3 c, dim3 d) {
}

// CHECK: void test(const sycl::range<3>& a, const sycl::range<3>& b) {
void test(const dim3& a, const dim3& b) {
}

// CHECK: void test(sycl::range<3>&& a, sycl::range<3>&& b) {
void test(dim3&& a, dim3&& b) {
}

// CHECK: void test(const sycl::range<3>* a, const sycl::range<3>* b) {
void test(const dim3* a, const dim3* b) {
}

// CHECK: void test(const sycl::range<3>** a, const sycl::range<3>** b) {
void test(const dim3** a, const dim3** b) {
}

__global__ void kernel(int dim) {}

int main() {
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
  // range default constructor does the right thing.
  // CHECK: sycl::range<3> deflt(1, 1, 1);
  dim3 deflt;

  // CHECK: sycl::range<3> round1(1, 1, 1);
  dim3 round1(1);
  // CHECK: sycl::range<3> round1_1(1, 1, NUM);
  dim3 round1_1(NUM);

  // CHECK: sycl::range<3> round2(1, 1, 2);
  dim3 round2(2, 1);
  // CHECK: sycl::range<3> round2_1(1, NUM, NUM);
  dim3 round2_1(NUM, NUM);

  // CHECK: sycl::range<3> assign = sycl::range<3>(1, 1, 32);
  dim3 assign = 32;
  // CHECK: sycl::range<3> assign_1 = sycl::range<3>(1, 1, NUM);
  dim3 assign_1 = NUM;

  // CHECK: sycl::range<3> castini = sycl::range<3>(1, 1, 4);
  dim3 castini = (dim3)4;
  // CHECK: sycl::range<3> castini_1 = sycl::range<3>(1, 1, NUM);
  dim3 castini_1 = (dim3)NUM;

  // CHECK: sycl::range<3> castini2 = sycl::range<3>(1, 2, 2);
  dim3 castini2 = dim3(2, 2);
  // CHECK: sycl::range<3> castini2_1 = sycl::range<3>(1, NUM, NUM);
  dim3 castini2_1 = dim3(NUM, NUM);

  // CHECK: sycl::range<3> castini3 = sycl::range<3>(10, 1, 3);
  dim3 castini3 = dim3(3, 1, 10);
  // CHECK: sycl::range<3> castini3_1 = sycl::range<3>(NUM, NUM, NUM);
  dim3 castini3_1 = dim3(NUM, NUM, NUM);

  // CHECK: deflt = sycl::range<3>(1, 1, 3);
  deflt = dim3(3);
  // CHECK: deflt = sycl::range<3>(1, 1, NUM);
  deflt = dim3(NUM);
  // CHECK: deflt = sycl::range<3>(1, 1, 5);
  deflt = 5;
  // CHECK: deflt = sycl::range<3>(1, 1, ((NUM%32 == 0) ? NUM/32 : (NUM/32 + 1)));
  deflt = ((NUM%32 == 0) ? NUM/32 : (NUM/32 + 1));

  // CHECK: sycl::range<3> copyctor1 = sycl::range<3>(sycl::range<3>(1, 1, 33));
  dim3 copyctor1 = dim3((dim3)33);
  // CHECK: sycl::range<3> copyctor1_1 = sycl::range<3>(sycl::range<3>(1, 1, NUM));
  dim3 copyctor1_1 = dim3((dim3)NUM);

  // CHECK: sycl::range<3> copyctor2 = sycl::range<3>(copyctor1);
  dim3 copyctor2 = dim3(copyctor1);

  // CHECK: sycl::range<3> copyctor3(copyctor1);
  dim3 copyctor3(copyctor1);

  // CHECK: func(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 2), sycl::range<3>(1, 2, 3));
  func((dim3)1, dim3(1), dim3(2, 1), dim3(3, 2, 1));
  // CHECK: func(sycl::range<3>(1, 1, NUM), sycl::range<3>(1, 1, NUM), sycl::range<3>(1, NUM, NUM), sycl::range<3>(NUM, NUM, NUM));
  func((dim3)NUM, dim3(NUM), dim3(NUM, NUM), dim3(NUM, NUM, NUM));
  // CHECK: func(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 2), sycl::range<3>(1, 1, 3), sycl::range<3>(1, 1, 4));
  func(1, 2, 3, 4);
  // CHECK: func(sycl::range<3>(1, 1, NUM), sycl::range<3>(1, 1, NUM), sycl::range<3>(1, 1, NUM), sycl::range<3>(1, 1, NUM));
  func(NUM, NUM, NUM, NUM);
  // CHECK: func(deflt, sycl::range<3>(deflt), sycl::range<3>(deflt), sycl::range<3>(1, 1, 2 + 3 * 3));
  func(deflt, dim3(deflt), (dim3)deflt, 2 + 3 * 3);
  // CHECK: func(deflt, sycl::range<3>(deflt), sycl::range<3>(deflt), sycl::range<3>(1, 1, NUM + NUM * NUM));
  func(deflt, dim3(deflt), (dim3)deflt, NUM + NUM * NUM);

  // CHECK: sycl::range<3> test(3, 2, 1);
  dim3 test(1, 2, 3);
  // CHECK: sycl::range<3> test_1(NUM, NUM, NUM);
  dim3 test_1(NUM, NUM, NUM);

  // CHECK: int b = test[2] + test[1] + test [0];
  int b = test.x + test. y + test .z;
  // CHECK: sycl::range<3> *p = &test;
  dim3 *p = &test;
  // CHECK: sycl::range<3> **pp = &p;
  dim3 **pp = &p;

  // CHECK: int a = (*p)[2] + (*p)[1] + (*p)[0];
  int a = p->x + p->y + p->z;
  // CHECK: int aa = (*(*pp))[2] + (*(*pp))[1] + (*(*pp))[0];
  int aa = (*pp)->x + (*pp)->y + (*pp)->z;

  struct  container
  {
    unsigned int x, y, z;
    // CHECK: sycl::range<3> w;
    dim3 w;
    // CHECK: sycl::range<3> *pw;
    dim3 *pw;
    // CHECK: sycl::range<3> **ppw;
    dim3 **ppw;
  };
  typedef  struct container container;

  container t;

  // CHECK: int c = t.w[2] + t.w[1] + t.w[0];
  int c = t.w.x + t.w.y + t.w.z;
  // CHECK: int c2 = (*t.pw)[2] + (*t.pw)[1] + (*t.pw)[0];
  int c2 = t.pw->x + t.pw->y + t.pw->z;
  // CHECK: int c3 = (*(*t.ppw))[2] + (*(*t.ppw))[1] + (*(*t.ppw))[0];
  int c3 = (*t.ppw)->x + (*t.ppw)->y + (*t.ppw)->z;

  // CHECK: sycl::range<3> d3_1(1, 1, test[2]);
  dim3 d3_1(test.x);
  // CHECK: sycl::range<3> d3_2(1, 1, test[2] + 1);
  dim3 d3_2(test.x + 1);
  // CHECK: sycl::range<3> d3_2_1(1, 1, static_cast<unsigned>(test[2] + 32));
  dim3 d3_2_1(static_cast<unsigned>(test.x + 32));
  // CHECK: sycl::range<3> d3_2_2(1, 1, test[2] + NUM);
  dim3 d3_2_2(test.x + NUM);
  // CHECK: sycl::range<3> d3_3(1, 1, 2 + test[2] + 1);
  dim3 d3_3(2 + test.x + 1);
  // CHECK: sycl::range<3> d3_3_1(1, 1, 32 + test[2] + 64);
  dim3 d3_3_1(32 + test.x + 64);
  // CHECK: sycl::range<3> d3_3_2(1, 1, NUM + test[2] + NUM);
  dim3 d3_3_2(NUM + test.x + NUM);
  // CHECK: sycl::range<3> d3_4(1, test[1], test[2]);
  dim3 d3_4(test.x, test.y);
  // CHECK: sycl::range<3> d3_5(test[0], test[1], test[2]);
  dim3 d3_5(test.x, test.y, test.z);
  // CHECK: sycl::range<3> d3_6 = sycl::range<3>(3 + test[0] + 4, 2 + test[1], test[2] + 1);
  dim3 d3_6 = dim3(test.x + 1, 2 + test.y, 3 + test.z + 4);
  // CHECK: sycl::range<3> d3_6_1 = sycl::range<3>(113 + test[0] + 114, 112 + test[1], test[2] + 111);
  dim3 d3_6_1 = dim3(test.x + 111, 112 + test.y, 113 + test.z + 114);
  // CHECK: sycl::range<3> d3_6_2 = sycl::range<3>(NUM + test[0] + NUM, NUM + test[1], test[2] + NUM);
  dim3 d3_6_2 = dim3(test.x + NUM, NUM + test.y, NUM + test.z + NUM);
  // todoCHECK: sycl::range<3> d3_6_3 = sycl::range<3>(NUM + test[0] + NUM, NUM + test[1], sycl::ceil(test[2] + NUM));
  dim3 d3_6_3 = dim3(ceil(test.x + NUM), NUM + test.y, NUM + test.z + NUM);
  // CHECK: sycl::range<3> gpu_blocks(1, 1, 1 / (d3_6_3[2] * 200));
  dim3 gpu_blocks(1 / (d3_6_3.x * 200));
  // CHECK:   q_ct1.parallel_for<dpct_kernel_name<class kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           kernel(d3_6[2]);
  // CHECK-NEXT:         });
  kernel<<<1, 1>>>(d3_6.x);
  // CHECK:   q_ct1.parallel_for<dpct_kernel_name<class kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, NUM) * sycl::range<3>(1, 1, NUM), sycl::range<3>(1, 1, NUM)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           kernel(d3_6[2]);
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
  // CHECK: sycl::range<3> dimBlock(1, 1, 256);
  // CHECK: sycl::range<3> dimGrid_2(1, 1, std::max(2048, 3));
  // CHECK: sycl::range<3> dimGrid_1(1, 1, std::max(2048, 3));
  // CHECK: std::min(2048, getgriddim<int>(n, dimBlock[2]));
  // CHECK: sycl::range<3> dimGrid(1, 1, std::min(2048, getgriddim<int>(n, dimBlock[2])));
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


// CHECK: void kernel_foo(float *a, wrap *mt, unsigned int N, sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:   const unsigned int i = item_ct1.get_group(2)*item_ct1.get_local_range(2)+item_ct1.get_local_id(2);
// CHECK-NEXT:   if (i<N) {
// CHECK-NEXT:     dpct::atomic_fetch_add(&mt[i].f3.x(), a[i]);
// CHECK-NEXT:   }
// CHECK-NEXT: }
__global__ void kernel_foo(float *a, wrap *mt, unsigned int N) {
  const unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i<N) {
    atomicAdd(&mt[i].f3.x, a[i]);
  }
}

// CHECK: void dim3_foo() {
// CHECK-NEXT:     DIM3_DEFAULT_VAR(block0(1, 1, 1));
// CHECK-NEXT:     CALL_FUNC( []() {
// CHECK-NEXT:         sycl::range<3> block1(1, 1, 1);
// CHECK-NEXT:         sycl::range<3> block2{1, 1, 1};
// CHECK-NEXT:         sycl::range<3> block3(1, 1, 2);
// CHECK-NEXT:         sycl::range<3> block4(1, 3, 2);
// CHECK-NEXT:         sycl::range<3> block5(4, 3, 2);
// CHECK-NEXT:         DIM3_DEFAULT_VAR(block6(1, 1, 1));
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
