// clang-format off
// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0
// RUN: dpct --format-range=none --use-experimental-features=matrix -out-root %T/wmma %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/wmma/wmma.dp.cpp --match-full-lines %s
#include <assert.h>
#include <cuda.h>
#include <iostream>
#include <mma.h>

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

  // Declare the fragments
  // CHECK: sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, sycl::half, sycl::ext::oneapi::experimental::matrix::use::a, WMMA_M, WMMA_K, sycl::ext::oneapi::experimental::matrix::layout::row_major>
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major>
      a_frag;
  // CHECK: sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, sycl::half, sycl::ext::oneapi::experimental::matrix::use::b, WMMA_N, WMMA_K, sycl::ext::oneapi::experimental::matrix::layout::col_major>
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major>
      b_frag;
  // CHECK: sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, float, sycl::ext::oneapi::experimental::matrix::use::accumulator, WMMA_M, WMMA_N> acc_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
  // CHECK: sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, float, sycl::ext::oneapi::experimental::matrix::use::accumulator, WMMA_M, WMMA_N> c_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  // CHECK: sycl::ext::oneapi::experimental::matrix::joint_matrix_fill(item_ct1.get_sub_group(), acc_frag, 0.0f);
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
      // CHECK: sycl::ext::oneapi::experimental::matrix::joint_matrix_load(item_ct1.get_sub_group(), a_frag, sycl::address_space_cast<sycl::access::address_space::generic_space, sycl::access::decorated::no, const sycl::half>(a + aCol + aRow * lda), lda);
      nvcuda::wmma::load_matrix_sync(a_frag, a + aCol + aRow * lda, lda);
      // CHECK: sycl::ext::oneapi::experimental::matrix::joint_matrix_load(item_ct1.get_sub_group(), b_frag, sycl::address_space_cast<sycl::access::address_space::generic_space, sycl::access::decorated::no, const sycl::half>(b + bRow + bCol * ldb), ldb);
      nvcuda::wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

      // Perform the matrix multiplication
      // CHECK: sycl::ext::oneapi::experimental::matrix::joint_matrix_mad(item_ct1.get_sub_group(), acc_frag, a_frag, b_frag, acc_frag);
      nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
  }

  // Load in the current value of c, scale it by beta, and add this our result
  // scaled by alpha
  int cCol = warpN * WMMA_N;
  int cRow = warpM * WMMA_M;

  if (cRow < m_ld && cCol < n_ld) {
    // CHECK: sycl::ext::oneapi::experimental::matrix::joint_matrix_load(item_ct1.get_sub_group(), c_frag, sycl::address_space_cast<sycl::access::address_space::generic_space, sycl::access::decorated::no, const float>(c + cCol + cRow * ldc), ldc, sycl::ext::oneapi::experimental::matrix::layout::row_major);
    nvcuda::wmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc,
                                   nvcuda::wmma::mem_row_major);

    // Store the output
    // CHECK: sycl::ext::oneapi::experimental::matrix::joint_matrix_store(item_ct1.get_sub_group(), c_frag, sycl::address_space_cast<sycl::access::address_space::generic_space, sycl::access::decorated::no, float>(d + cCol + cRow * ldc), ldc, sycl::ext::oneapi::experimental::matrix::layout::col_major);
    nvcuda::wmma::store_matrix_sync(d + cCol + cRow * ldc, c_frag, ldc,
                                    nvcuda::wmma::mem_col_major);
  }
}

int main() {
  half *A_h = NULL;
  half *B_h = NULL;
  float *C_h = NULL;
  A_h = (half *)malloc(sizeof(half) * M_GLOBAL * K_GLOBAL);
  B_h = (half *)malloc(sizeof(half) * K_GLOBAL * N_GLOBAL);
  C_h = (float *)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);
  half *A = NULL;
  half *B = NULL;
  float *C = NULL;
  float *D = NULL;

  cudaMalloc(reinterpret_cast<void **>(&A),
             sizeof(half) * M_GLOBAL * K_GLOBAL);
  cudaMalloc(reinterpret_cast<void **>(&B),
             sizeof(half) * N_GLOBAL * K_GLOBAL);
  cudaMalloc(reinterpret_cast<void **>(&C),
             sizeof(float) * M_GLOBAL * N_GLOBAL);
  cudaMalloc(reinterpret_cast<void **>(&D),
             sizeof(float) * M_GLOBAL * N_GLOBAL);

  assert(((unsigned long long)A) % 128 == 0);
  assert(((unsigned long long)B) % 128 == 0);
  assert(((unsigned long long)C) % 128 == 0);
  assert(((unsigned long long)D) % 128 == 0);

  init_host_matrices(A_h, B_h, C_h);

  printf("Preparing data for GPU...\n");

  cudaMemcpy(A, A_h, sizeof(half) * M_GLOBAL * K_GLOBAL,
             cudaMemcpyHostToDevice);
  cudaMemcpy(B, B_h, sizeof(half) * N_GLOBAL * K_GLOBAL,
             cudaMemcpyHostToDevice);
  cudaMemcpy(C, C_h, sizeof(float) * M_GLOBAL * N_GLOBAL,
             cudaMemcpyHostToDevice);
  cudaMemset(D, 0, sizeof(float) * M_GLOBAL * N_GLOBAL);

  const float alpha = 1.1f;
  const float beta = 1.2f;

  dim3 gridDim;
  dim3 blockDim;

  // blockDim.x must be a multple of warpSize
  // 128x4 means we have 16 warps and a block computes a 64x64 output tile
  blockDim.x = 128;
  blockDim.y = 4;

  gridDim.x = (M_GLOBAL + (WMMA_M * blockDim.x / 32 - 1)) /
              (WMMA_M * blockDim.x / 32);
  gridDim.y = (N_GLOBAL + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

  printf("Computing... using simple_wmma_gemm kernel\n");
  simple_wmma_gemm<<<gridDim, blockDim>>>(A, B, C, D, M_GLOBAL, N_GLOBAL,
                                          K_GLOBAL, alpha, beta);
  cudaDeviceSynchronize();

  free(A_h);
  free(B_h);
  free(C_h);
  cudaFree(reinterpret_cast<void *>(A));
  cudaFree(reinterpret_cast<void *>(B));
  cudaFree(reinterpret_cast<void *>(C));
  cudaFree(reinterpret_cast<void *>(D));

  return 0;
}
// clang-format on
