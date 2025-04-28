// RUN: dpct --format-range=none -out-root %T/iteration-space %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/iteration-space/iteration-space.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/iteration-space/iteration-space.dp.cpp -o %T/iteration-space/iteration-space.dp.o %}

// Test that the replacement happens when it should to.
// CHECK: void test_00() {
__global__
void test_00() {
  // CHECK: auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  // CHECK: size_t tix = item_ct1.get_local_id(2);
  // CHECK: size_t tiy = item_ct1.get_local_id(1);
  // CHECK: size_t tiz = item_ct1.get_local_id(0);
  size_t tix = threadIdx.x;
  size_t tiy = threadIdx.y;
  size_t tiz = threadIdx.z;

  // CHECK: size_t bix = item_ct1.get_group(2);
  // CHECK: size_t biy = item_ct1.get_group(1);
  // CHECK: size_t biz = item_ct1.get_group(0);

  size_t bix = blockIdx.x;
  size_t biy = blockIdx.y;
  size_t biz = blockIdx.z;

  // CHECK: size_t bdx = item_ct1.get_local_range(2);
  // CHECK: size_t bdy = item_ct1.get_local_range(1);
  // CHECK: size_t bdz = item_ct1.get_local_range(0);
  size_t bdx = blockDim.x;
  size_t bdy = blockDim.y;
  size_t bdz = blockDim.z;

  // CHECK: size_t gdx = item_ct1.get_group_range(2);
  // CHECK: size_t gdy = item_ct1.get_group_range(1);
  // CHECK: size_t gdz = item_ct1.get_group_range(0);

  size_t gdx = gridDim.x;
  size_t gdy = gridDim.y;
  size_t gdz = gridDim.z;
}

// Test that the replacement doesn't happen in host functions.
// CHECK: void test_01() {
void test_01() {
  uint3 threadIdx, blockIdx, blockDim, gridDim;

  // CHECK:size_t tix = threadIdx.x();
  // CHECK:size_t tiy = threadIdx.y();
  // CHECK:size_t tiz = threadIdx.z();
  size_t tix = threadIdx.x;
  size_t tiy = threadIdx.y;
  size_t tiz = threadIdx.z;

  // CHECK: size_t bix = blockIdx.x();
  // CHECK: size_t biy = blockIdx.y();
  // CHECK: size_t biz = blockIdx.z();
  size_t bix = blockIdx.x;
  size_t biy = blockIdx.y;
  size_t biz = blockIdx.z;

  // CHECK:  size_t bdx = blockDim.x();
  // CHECK:  size_t bdy = blockDim.y();
  // CHECK:  size_t bdz = blockDim.z();
  size_t bdx = blockDim.x;
  size_t bdy = blockDim.y;
  size_t bdz = blockDim.z;

  // CHECK:  size_t gdx = gridDim.x();
  // CHECK:  size_t gdy = gridDim.y();
  // CHECK:  size_t gdz = gridDim.z();
  size_t gdx = gridDim.x;
  size_t gdy = gridDim.y;
  size_t gdz = gridDim.z;
}

// Test that the replacement doesn't happen if threadIdx is redefined.
// CHECK: void test_02() {
__global__ void test_02() {
  uint3 threadIdx, blockIdx, blockDim, gridDim;

  // CHECK:size_t tix = threadIdx.x();
  // CHECK:size_t tiy = threadIdx.y();
  // CHECK:size_t tiz = threadIdx.z();
  size_t tix = threadIdx.x;
  size_t tiy = threadIdx.y;
  size_t tiz = threadIdx.z;

  // CHECK: size_t bix = blockIdx.x();
  // CHECK: size_t biy = blockIdx.y();
  // CHECK: size_t biz = blockIdx.z();
  size_t bix = blockIdx.x;
  size_t biy = blockIdx.y;
  size_t biz = blockIdx.z;

  // CHECK:  size_t bdx = blockDim.x();
  // CHECK:  size_t bdy = blockDim.y();
  // CHECK:  size_t bdz = blockDim.z();
  size_t bdx = blockDim.x;
  size_t bdy = blockDim.y;
  size_t bdz = blockDim.z;

  // CHECK:  size_t gdx = gridDim.x();
  // CHECK:  size_t gdy = gridDim.y();
  // CHECK:  size_t gdz = gridDim.z();
  size_t gdx = gridDim.x;
  size_t gdy = gridDim.y;
  size_t gdz = gridDim.z;
}

