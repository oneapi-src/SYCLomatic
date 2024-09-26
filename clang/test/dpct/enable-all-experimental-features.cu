// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5
// RUN: dpct --format-range=none -out-root %T/enable-all-experimental-features %s --cuda-include-path="%cuda-path/include" --use-experimental-features=all
// RUN: FileCheck --input-file %T/enable-all-experimental-features/enable-all-experimental-features.dp.cpp --match-full-lines %s

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <mma.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>

// free-function-queries

namespace cg = cooperative_groups;

// CHECK:void testThreadGroup(dpct::experimental::group_base<3> g) {
__device__ void testThreadGroup(cg::thread_group g) {
  // CHECK:  g.get_local_linear_id();
  g.thread_rank();
  // CHECK:  g.barrier();
  g.sync();
  // CHECK:  g.get_local_linear_range();
  g.size();

  auto block = cg::this_thread_block();
  // CHECK: dpct::dim3(block.get_local_id());
  block.thread_index();
}

__global__ void kernelFunc() {
  auto block = cg::this_thread_block();
  // CHECK: dpct::dim3(block.get_local_id());
  block.thread_index();
  // CHECK:  auto threadBlockGroup = sycl::ext::oneapi::this_work_item::get_work_group<3>();
  auto threadBlockGroup = cg::this_thread_block();

  // CHECK:   testThreadGroup(dpct::experimental::group(threadBlockGroup, item_ct1));
  testThreadGroup(threadBlockGroup);
  // CHECK:  dpct::experimental::logical_group tilePartition16 = dpct::experimental::logical_group(item_ct1, sycl::ext::oneapi::this_work_item::get_work_group<3>(), 16);
  cg::thread_block_tile<16> tilePartition16 = cg::tiled_partition<16>(threadBlockGroup);
  // CHECK:   testThreadGroup(dpct::experimental::group(tilePartition16, item_ct1));
  testThreadGroup(tilePartition16);
  // CHECK:  sycl::sub_group tilePartition32 = sycl::ext::oneapi::this_work_item::get_sub_group();
  cg::thread_block_tile<32> tilePartition32 = cg::tiled_partition<32>(threadBlockGroup);
  // CHECK:   testThreadGroup(dpct::experimental::group(tilePartition32, item_ct1));
  testThreadGroup(tilePartition32);
  // CHECK:  dpct::experimental::logical_group tilePartition16_1(dpct::experimental::logical_group(item_ct1, sycl::ext::oneapi::this_work_item::get_work_group<3>(), 16));
  // CHECK:  sycl::sub_group tilePartition32_2(sycl::ext::oneapi::this_work_item::get_sub_group());
  cg::thread_block_tile<16> tilePartition16_1(cg::tiled_partition<16>(threadBlockGroup));
  cg::thread_block_tile<32> tilePartition32_2(cg::tiled_partition<32>(threadBlockGroup));
}
// local-memory-kernel-scope-allocation

class TestObject {
public:
  // CHECK: static void run(int *in, int *out) {
  // CHECK-NEXT:    /*
  // CHECK-NEXT:    DPCT1115:{{[0-9]+}}: The sycl::ext::oneapi::group_local_memory_for_overwrite is used to allocate group-local memory at the none kernel functor scope of a work-group data parallel kernel. You may need to adjust the code.
  // CHECK-NEXT:    */
  // CHECK-NEXT:  auto &a0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(sycl::ext::oneapi::this_work_item::get_work_group<3>()); // the size of s is static
  // CHECK-NEXT:  a0 = sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2);
  __device__ static void run(int *in, int *out) {
    __shared__ int a0; // the size of s is static
    a0 = threadIdx.x;
  }
  __device__ void test() {}
};

// CHECK: void memberAcc() {
// CHECK-NEXT: auto &s = *sycl::ext::oneapi::group_local_memory_for_overwrite<TestObject>(sycl::ext::oneapi::this_work_item::get_work_group<3>()); // the size of s is static
// CHECK-NEXT: s.test();
// CHECK-NEXT: }
__global__ void memberAcc() {
  __shared__ TestObject s; // the size of s is static
  s.test();
}

// logical-group

// CHECK:  #define test33(a) testThreadGroup(a)
#define test33(a) testThreadGroup(a)
#define test22(a) test33(a)
#define test11(a) test22(a)
// CHECK:  #define test44(a) testThreadGroup(a)
#define test44(a) testThreadGroup(a)

namespace cg = cooperative_groups;

__global__ void kernelFunc1() {
  auto block = cg::this_thread_block();
  // CHECK: dpct::dim3(block.get_local_id());
  block.thread_index();
  // CHECK:  auto threadBlockGroup = sycl::ext::oneapi::this_work_item::get_work_group<3>();
  auto threadBlockGroup = cg::this_thread_block();

  testThreadGroup(threadBlockGroup);

  cg::thread_block_tile<16> tilePartition16 = cg::tiled_partition<16>(threadBlockGroup);
  testThreadGroup(tilePartition16);
  cg::thread_block_tile<32> tilePartition32 = cg::tiled_partition<32>(threadBlockGroup);
  testThreadGroup(tilePartition32);
  cg::thread_block_tile<16> tilePartition16_1(cg::tiled_partition<16>(threadBlockGroup));
  cg::thread_block_tile<32> tilePartition32_2(cg::tiled_partition<32>(threadBlockGroup));

  test11(tilePartition16);
  testThreadGroup(tilePartition16);
  // CHECK:   test44(dpct::experimental::group(tilePartition16, sycl::ext::oneapi::this_work_item::get_nd_item<3>()));
  // CHECK:   test11(dpct::experimental::group(tilePartition32, sycl::ext::oneapi::this_work_item::get_nd_item<3>()));
  // CHECK:   test11(dpct::experimental::group(threadBlockGroup, sycl::ext::oneapi::this_work_item::get_nd_item<3>()));
  test44(tilePartition16);
  test11(tilePartition32);
  test11(threadBlockGroup);
}

// sycl::group_barrier(root_group)

// CHECK: void kernel(sycl::atomic_ref<unsigned int, sycl::memory_order::seq_cst, sycl::memory_scope::device, sycl::access::address_space::global_space> &sync_ct1) {
// CHECK-NEXT:  sycl::group_barrier(grid);
// CHECK-NEXT: }
__global__ void kernel() {
  cg::grid_group grid = cg::this_grid();
  grid.sync();
}

#define DATA_NUM 100

template <typename T = int>
void init_data(T *data, int num) {
  for (int i = 0; i < num; i++)
    data[i] = i;
}

template <typename T = int>
bool verify_data(T *data, T *expect, int num, int step = 1) {
  for (int i = 0; i < num; i = i + step) {
    if (data[i] != expect[i]) {
      return false;
    }
  }
  return true;
}

template <typename T = int>
void print_data(T *data, int num) {
  for (int i = 0; i < num; i++) {
    std::cout << data[i] << ", ";
    if ((i + 1) % 32 == 0)
      std::cout << std::endl;
  }
  std::cout << std::endl;
}

struct UserMin {
  template <typename T>
  __device__ __host__ __forceinline__
      T
      operator()(const T &a, const T &b) const {
    return (b < a) ? b : a;
  }
};

bool test_reduce_1() {
  int num_segments = 10;
  int *device_offsets;
  int *device_in;
  int *device_out;
  UserMin min_op;
  int initial_value = INT_MAX;
  void *temp_storage = NULL;
  size_t temp_storage_size = 0;
  int expect[DATA_NUM] = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90};

  cudaMallocManaged(&device_offsets, (num_segments + 1) * sizeof(int));
  cudaMallocManaged(&device_in, DATA_NUM * sizeof(int));
  cudaMallocManaged(&device_out, num_segments * sizeof(int));
  init_data(device_in, DATA_NUM);
  for (int i = 0; i < num_segments + 1; i++) {
    device_offsets[i] = i * 10;
  }
  cub::DeviceSegmentedReduce::Reduce(temp_storage, temp_storage_size, device_in, device_out, num_segments, device_offsets, device_offsets + 1, min_op, initial_value);

  cudaMalloc(&temp_storage, temp_storage_size);

  // CHECK: dpct::device::experimental::segmented_reduce<128>(q_ct1, device_in, device_out, num_segments, device_offsets, device_offsets + 1, min_op, initial_value);
  cub::DeviceSegmentedReduce::Reduce(temp_storage, temp_storage_size, device_in, device_out, num_segments, device_offsets, device_offsets + 1, min_op, initial_value);

  cudaDeviceSynchronize();

  if (!verify_data(device_out, expect, num_segments)) {
    std::cout << "Reduce"
              << " verify failed" << std::endl;
    std::cout << "expect:" << std::endl;
    print_data<int>(expect, num_segments);
    std::cout << "current result:" << std::endl;
    print_data<int>(device_out, num_segments);
    return false;
  }
  return true;
}

// masked-sub-group-operation

__global__ void kernel1() {
  unsigned mask;
  int val;
  int srcLane;
  // CHECK: /*
  // CHECK: DPCT1108:{{[0-9]+}}: '__shfl_sync' was migrated with the experimental feature masked sub_group function which may not be supported by all compilers or runtimes. You may need to adjust the code.
  // CHECK: */
  // CHECK: dpct::experimental::select_from_sub_group(mask, sycl::ext::oneapi::this_work_item::get_sub_group(), val, srcLane);
  __shfl_sync(mask, val, srcLane);
}

// dpl-experimental-api

int thrust1() {
  // CHECK: dpct::device_vector<float> dVec(4);
  thrust::device_vector<float> dVec(4);
  // CHECK: auto loop_body = [=] (int ind) -> void {};
  auto loop_body = [=] __device__ __host__(int ind) -> void {};

  // CHECK: oneapi::dpl::experimental::for_each_async(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), dVec.begin(), dVec.end(), loop_body);
  thrust::for_each(thrust::cuda::par_nosync, dVec.begin(), dVec.end(), loop_body);
  return 0;
}

// occupancy-calculation

__global__ void k() {}

int occupancy() {
  int num_blocks;
  int block_size = 128;
  size_t dynamic_shared_memory_size = 0;
  // CHECK: /*
  // CHECK: DPCT1111:{{[0-9]+}}: Please verify the input arguments of "dpct::experimental::calculate_max_active_wg_per_xecore" base on the target function "k".
  // CHECK: */
  // CHECK: dpct::experimental::calculate_max_active_wg_per_xecore(&num_blocks, block_size, dynamic_shared_memory_size + dpct_placeholder /* total share local memory size */);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, k, block_size, dynamic_shared_memory_size);

  CUfunction func;
  // CHECK: /*
  // CHECK: DPCT1111:{{[0-9]+}}: Please verify the input arguments of "dpct::experimental::calculate_max_active_wg_per_xecore" base on the target function "func".
  // CHECK: */
  // CHECK: dpct::experimental::calculate_max_active_wg_per_xecore(&num_blocks, block_size, dynamic_shared_memory_size + dpct_placeholder /* total share local memory size */);
  cuOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, func, block_size, dynamic_shared_memory_size);

  int min_grid_size;
  int block_size_limit;
  // CHECK: /*
  // CHECK-NEXT: DPCT1111:{{[0-9]+}}: Please verify the input arguments of "dpct::experimental::calculate_max_potential_wg" base on the target function "k".
  // CHECK-NEXT: */
  // CHECK-NEXT:dpct::experimental::calculate_max_potential_wg(&min_grid_size, &block_size, 0, dpct_placeholder /* total share local memory size */);
  cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, k);
  // CHECK: /*
  // CHECK-NEXT: DPCT1111:{{[0-9]+}}: Please verify the input arguments of "dpct::experimental::calculate_max_potential_wg" base on the target function "k".
  // CHECK-NEXT: */
  // CHECK-NEXT:dpct::experimental::calculate_max_potential_wg(&min_grid_size, &block_size, 0, dynamic_shared_memory_size + dpct_placeholder /* total share local memory size */);
  cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, k, dynamic_shared_memory_size);
  // CHECK: /*
  // CHECK-NEXT: DPCT1111:{{[0-9]+}}: Please verify the input arguments of "dpct::experimental::calculate_max_potential_wg" base on the target function "k".
  // CHECK-NEXT: */
  // CHECK-NEXT:dpct::experimental::calculate_max_potential_wg(&min_grid_size, &block_size, block_size_limit, dynamic_shared_memory_size + dpct_placeholder /* total share local memory size */);
  cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, k, dynamic_shared_memory_size, block_size_limit);
  return 0;
}

// matrix

#define WARP_SIZE 32

// MMA matrix tile dimensions.

#define M 16
#define N 16
#define K 16

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define M_TILES 16
#define N_TILES 16
#define K_TILES 16

#define M_GLOBAL (M * M_TILES)
#define N_GLOBAL (N * N_TILES)
#define K_GLOBAL (K * K_TILES)

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4

#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2

#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)

__host__ void init_host_matrices(half *a, half *b, float *c) {
  for (int i = 0; i < M_GLOBAL; i++) {
    for (int j = 0; j < K_GLOBAL; j++) {
      a[i * K_GLOBAL + j] = (half)(rand() % 3);
    }
  }

  for (int i = 0; i < N_GLOBAL; i++) {
    for (int j = 0; j < K_GLOBAL; j++) {
      b[i * K_GLOBAL + j] = (half)(rand() % 3);
    }
  }

  for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++) {
    c[t] = static_cast<float>(rand() % 3);
  }
}

__global__ void simple_wmma_gemm(half *a, half *b, float *c, float *d, int m_ld,
                                 int n_ld, int k_ld, float alpha, float beta) {
  // Leading dimensions. Packed with no transpositions.
  int lda = k_ld;
  int ldb = k_ld;
  int ldc = n_ld;

  // Tile using a 2D grid
  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
  // CHECK: sycl::ext::oneapi::experimental::matrix::layout ly = sycl::ext::oneapi::experimental::matrix::layout::row_major;
  nvcuda::wmma::layout_t ly = nvcuda::wmma::mem_row_major;
  // Declare the fragments
  // CHECK: dpct::experimental::matrix::joint_matrix<dpct::experimental::matrix::a, WMMA_M, WMMA_N, WMMA_K, sycl::half, dpct::experimental::matrix::row_major>
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major>
      a_frag;
  // CHECK: dpct::experimental::matrix::joint_matrix<dpct::experimental::matrix::b, WMMA_M, WMMA_N, WMMA_K, sycl::half, dpct::experimental::matrix::col_major>
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major>
      b_frag;
  // CHECK: dpct::experimental::matrix::joint_matrix<dpct::experimental::matrix::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
  // CHECK: dpct::experimental::matrix::joint_matrix<dpct::experimental::matrix::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  // CHECK: sycl::ext::oneapi::experimental::matrix::joint_matrix_fill(sycl::ext::oneapi::this_work_item::get_sub_group(), acc_frag.get(), 0.0f);
  nvcuda::wmma::fill_fragment(acc_frag, 0.0f);

  // Loop over k
  for (int i = 0; i < k_ld; i += WMMA_K) {
    int aCol = i;
    int aRow = warpM * WMMA_M;
    int bCol = warpN * N;
    int bRow = i;

    // Bounds checking
    if (aRow < m_ld && aCol < k_ld && bRow < k_ld && bCol < n_ld) {
      // Load the inputs
      // CHECK: sycl::ext::oneapi::experimental::matrix::joint_matrix_load(sycl::ext::oneapi::this_work_item::get_sub_group(), a_frag.get(), sycl::address_space_cast<sycl::access::address_space::generic_space, sycl::access::decorated::no, const sycl::half>(a + aCol + aRow * lda), lda);
      nvcuda::wmma::load_matrix_sync(a_frag, a + aCol + aRow * lda, lda);
      // CHECK: sycl::ext::oneapi::experimental::matrix::joint_matrix_load(sycl::ext::oneapi::this_work_item::get_sub_group(), b_frag.get(), sycl::address_space_cast<sycl::access::address_space::generic_space, sycl::access::decorated::no, const sycl::half>(b + bRow + bCol * ldb), ldb);
      nvcuda::wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

      // Perform the matrix multiplication
      // CHECK: sycl::ext::oneapi::experimental::matrix::joint_matrix_mad(sycl::ext::oneapi::this_work_item::get_sub_group(), acc_frag.get(), a_frag.get(), b_frag.get(), acc_frag.get());
      nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
  }

  // Load in the current value of c, scale it by beta, and add this our result
  // scaled by alpha
  int cCol = warpN * WMMA_N;
  int cRow = warpM * WMMA_M;

  if (cRow < m_ld && cCol < n_ld) {
    // CHECK: sycl::ext::oneapi::experimental::matrix::joint_matrix_load(sycl::ext::oneapi::this_work_item::get_sub_group(), c_frag.get(), sycl::address_space_cast<sycl::access::address_space::generic_space, sycl::access::decorated::no, const float>(c + cCol + cRow * ldc), ldc, sycl::ext::oneapi::experimental::matrix::layout::row_major);
    nvcuda::wmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc, nvcuda::wmma::mem_row_major);
    // CHECK: sycl::ext::oneapi::experimental::matrix::joint_matrix_load(sycl::ext::oneapi::this_work_item::get_sub_group(), c_frag.get(), sycl::address_space_cast<sycl::access::address_space::generic_space, sycl::access::decorated::no, const float>(c + cCol + cRow * ldc), ldc, ly);
    nvcuda::wmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc, ly);
    // Store the output
    // CHECK: sycl::ext::oneapi::experimental::matrix::joint_matrix_store(sycl::ext::oneapi::this_work_item::get_sub_group(), c_frag.get(), sycl::address_space_cast<sycl::access::address_space::generic_space, sycl::access::decorated::no, float>(d + cCol + cRow * ldc), ldc, sycl::ext::oneapi::experimental::matrix::layout::col_major);
    nvcuda::wmma::store_matrix_sync(d + cCol + cRow * ldc, c_frag, ldc, nvcuda::wmma::mem_col_major);
  }
}

// bfloat16_math_functions

__global__ void kernelFuncBfloat16Arithmetic() {
  // CHECK: sycl::ext::oneapi::bfloat16 bf16, bf16_1, bf16_2, bf16_3;
  __nv_bfloat16 bf16, bf16_1, bf16_2, bf16_3;
  // CHECK: bf16 = sycl::ext::oneapi::experimental::fabs(bf16_1);
  bf16 = __habs(bf16_1);
  // CHECK: bf16 = bf16_1 + bf16_2;
  bf16 = __hadd(bf16_1, bf16_2);
  // CHECK: bf16 = bf16_1 + bf16_2;
  bf16 = __hadd_rn(bf16_1, bf16_2);
  // CHECK: bf16 = dpct::clamp<sycl::ext::oneapi::bfloat16>(bf16_1 + bf16_2, 0.f, 1.0f);
  bf16 = __hadd_sat(bf16_1, bf16_2);
  // CHECK: bf16 = bf16_1 / bf16_2;
  bf16 = __hdiv(bf16_1, bf16_2);
  // CHECK: bf16 = sycl::ext::oneapi::experimental::fma(bf16_1, bf16_2, bf16_3);
  bf16 = __hfma(bf16_1, bf16_2, bf16_3);
  // CHECK: bf16 = dpct::relu(sycl::ext::oneapi::experimental::fma(bf16_1, bf16_2, bf16_3));
  bf16 = __hfma_relu(bf16_1, bf16_2, bf16_3);
  // CHECK: bf16 = dpct::clamp<sycl::ext::oneapi::bfloat16>(sycl::ext::oneapi::experimental::fma(bf16_1, bf16_2, bf16_3), 0.f, 1.0f);
  bf16 = __hfma_sat(bf16_1, bf16_2, bf16_3);
  // CHECK: bf16 = bf16_1 * bf16_2;
  bf16 = __hmul(bf16_1, bf16_2);
  // CHECK: bf16 = bf16_1 * bf16_2;
  bf16 = __hmul_rn(bf16_1, bf16_2);
  // CHECK: bf16 = dpct::clamp<sycl::ext::oneapi::bfloat16>(bf16_1 * bf16_2, 0.f, 1.0f);
  bf16 = __hmul_sat(bf16_1, bf16_2);
  // CHECK: bf16 = -bf16_1;
  bf16 = __hneg(bf16_1);
  // CHECK: bf16 = bf16_1 - bf16_2;
  bf16 = __hsub(bf16_1, bf16_2);
  // CHECK: bf16 = bf16_1 - bf16_2;
  bf16 = __hsub_rn(bf16_1, bf16_2);
  // CHECK: bf16 = dpct::clamp<sycl::ext::oneapi::bfloat16>(bf16_1 - bf16_2, 0.f, 1.0f);
  bf16 = __hsub_sat(bf16_1, bf16_2);
}

// bindless_images
void driver() {
  // CHECK: sycl::ext::oneapi::experimental::sampled_image_handle o;
  CUtexObject o;
  // CHECK: dpct::image_data R;
  CUDA_RESOURCE_DESC R;
  // CHECK: dpct::sampling_info T;
  CUDA_TEXTURE_DESC T;
  // CHECK: o = dpct::experimental::create_bindless_image(R, T);
  cuTexObjectCreate(&o, &R, &T, NULL);
  // CHECK: dpct::experimental::destroy_bindless_image(o, dpct::get_in_order_queue());
  cuTexObjectDestroy(o);
  // CHECK: R = dpct::experimental::get_data(o);
  cuTexObjectGetResourceDesc(&R, o);
  // CHECK: T = dpct::experimental::get_sampling_info(o);
  cuTexObjectGetTextureDesc(&T, o);
}

// non-uniform-groups

__device__ void bar(int *arr, int *brr) {
  arr[threadIdx.x] = threadIdx.x + 10;
  if (threadIdx.x % 2 == 0) {
    for (int i = 0; i < 1000; ++i)
      arr[threadIdx.x] += arr[threadIdx.x] - 1 * arr[threadIdx.x] - 3;
    if (arr[threadIdx.x] < 0)
      arr[threadIdx.x] = 0;
  }

  // CHECK: sycl::group_barrier(sycl::ext::oneapi::experimental::get_ballot_group(item_ct1.get_sub_group(), 0b1010101010 & (1 << item_ct1.get_local_linear_id())));
  asm volatile("bar.warp.sync %0;" ::"r"(0b1010101010));
  if (threadIdx.x == 1) {
    for (int i = 0; i < 10; ++i) {
      brr[i] = arr[i];
    }
  }
}
