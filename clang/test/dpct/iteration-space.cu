// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/iteration-space.dp.cpp --match-full-lines %s

// Test that the replacement happens when it should to.
// CHECK: void test_00(sycl::nd_item<3> [[ITEMNAME:item_ct1]]) {
__global__
void test_00() {
  // CHECK: size_t tix = [[ITEMNAME]].get_local_id(2);
  // CHECK: size_t tiy = [[ITEMNAME]].get_local_id(1);
  // CHECK: size_t tiz = [[ITEMNAME]].get_local_id(0);
  size_t tix = threadIdx.x;
  size_t tiy = threadIdx.y;
  size_t tiz = threadIdx.z;

  // CHECK: size_t bix = [[ITEMNAME]].get_group(2);
  // CHECK: size_t biy = [[ITEMNAME]].get_group(1);
  // CHECK: size_t biz = [[ITEMNAME]].get_group(0);

  size_t bix = blockIdx.x;
  size_t biy = blockIdx.y;
  size_t biz = blockIdx.z;

  // CHECK: size_t bdx = [[ITEMNAME]].get_local_range().get(2);
  // CHECK: size_t bdy = [[ITEMNAME]].get_local_range().get(1);
  // CHECK: size_t bdz = [[ITEMNAME]].get_local_range().get(0);
  size_t bdx = blockDim.x;
  size_t bdy = blockDim.y;
  size_t bdz = blockDim.z;

  // CHECK: size_t gdx = [[ITEMNAME]].get_group_range(2);
  // CHECK: size_t gdy = [[ITEMNAME]].get_group_range(1);
  // CHECK: size_t gdz = [[ITEMNAME]].get_group_range(0);

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
