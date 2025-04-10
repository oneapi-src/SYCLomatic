// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// UNSUPPORTED: system-windows
// RUN: dpct --format-range=none -in-root %S -out-root %T/blocklevel/blockload %S/blockload.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/blocklevel/blockload/blockload.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/blocklevel/blockload/blockload.dp.cpp -o %T/blocklevel/blockload/blockload.dp.o %}

#include <cub/cub.cuh>
// CHECK: #include <dpct/group_utils.hpp>
__global__ void DirectKernel(int *d_data, int valid_items, int default_value) {
  int thread_data[4] = {0};
// CHECK:  dpct::group::load_direct_blocked(item_ct1, d_data, thread_data);
// CHECK:  dpct::group::load_direct_blocked(item_ct1, d_data, thread_data, valid_items);
// CHECK:  dpct::group::load_direct_blocked(item_ct1, d_data, thread_data, valid_items, default_value);
// CHECK:  dpct::group::load_direct_striped(item_ct1, d_data, thread_data);
// CHECK:  dpct::group::load_direct_striped(item_ct1, d_data, thread_data, valid_items);
// CHECK:  dpct::group::load_direct_striped(item_ct1, d_data, thread_data, valid_items, default_value);
// CHECK:  dpct::group::load_direct_sub_group_striped(item_ct1, d_data, thread_data);
// CHECK:  dpct::group::load_direct_sub_group_striped(item_ct1, d_data, thread_data, valid_items);
// CHECK:  dpct::group::load_direct_sub_group_striped(item_ct1, d_data, thread_data, valid_items, default_value);
  cub::LoadDirectBlocked(threadIdx.x, d_data, thread_data);
  cub::LoadDirectBlocked(threadIdx.x, d_data, thread_data, valid_items);
  cub::LoadDirectBlocked(threadIdx.x, d_data, thread_data, valid_items, default_value);
  cub::LoadDirectStriped<128>(threadIdx.x, d_data, thread_data);
  cub::LoadDirectStriped<128>(threadIdx.x, d_data, thread_data, valid_items);
  cub::LoadDirectStriped<128>(threadIdx.x, d_data, thread_data, valid_items, default_value);
  cub::LoadDirectWarpStriped(threadIdx.x, d_data, thread_data);
  cub::LoadDirectWarpStriped(threadIdx.x, d_data, thread_data, valid_items);
  cub::LoadDirectWarpStriped(threadIdx.x, d_data, thread_data, valid_items, default_value);
}

__global__ void BlockedKernel(int *d_data, int valid_items) {
  // Specialize BlockLoad for a 1D block of 128 threads owning 4 integer items each
  // CHECK: using BlockLoad = dpct::group::group_load<int, 4>;
  using BlockLoad = cub::BlockLoad<int, 128, 4>;
  __shared__ typename BlockLoad::TempStorage temp_storage;

  int thread_data[4] = {0};

  // CHECK: BlockLoad(temp_storage).load(item_ct1, d_data, thread_data, valid_items);
  BlockLoad(temp_storage).Load(d_data, thread_data, valid_items);
  printf("%3d: [%d, %d, %d, %d]\n", threadIdx.x, thread_data[0], thread_data[1], thread_data[2], thread_data[3]);
}

__global__ void StripedKernel(int *d_data, int valid_items) {
  // Specialize BlockLoad for a 1D block of 128 threads owning 4 integer items each
  // CHECK: using BlockLoad = dpct::group::group_load<int, 4, dpct::group::group_load_algorithm::striped>;
  using BlockLoad = cub::BlockLoad<int, 128, 4, cub::BLOCK_LOAD_STRIPED>;
  __shared__ typename BlockLoad::TempStorage temp_storage;

  int thread_data[4] = {0};
  // CHECK: BlockLoad(temp_storage).load(item_ct1, d_data, thread_data, valid_items);
  BlockLoad(temp_storage).Load(d_data, thread_data, valid_items);
  printf("%3d: [%d, %d, %d, %d]\n", threadIdx.x, thread_data[0], thread_data[1], thread_data[2], thread_data[3]);
}

__global__ void VecKernel(int *d_data, int valid_items) {
  // Specialize BlockLoad for a 1D block of 128 threads owning 4 integer items each
  // CHECK: using BlockLoad = dpct::group::group_load<int, 4, dpct::group::group_load_algorithm::blocked>;
  using BlockLoad = cub::BlockLoad<int, 128, 4, cub::BLOCK_LOAD_VECTORIZE>;
  __shared__ typename BlockLoad::TempStorage temp_storage;

  int thread_data[4] = {0};
  // CHECK: BlockLoad(temp_storage).load(item_ct1, d_data, thread_data, valid_items);
  BlockLoad(temp_storage).Load(d_data, thread_data, valid_items);
  printf("%3d: [%d, %d, %d, %d]\n", threadIdx.x, thread_data[0], thread_data[1], thread_data[2], thread_data[3]);
}

__global__ void TransposeKernel(int *d_data, int valid_items) {
  // Specialize BlockLoad for a 1D block of 128 threads owning 4 integer items each
  // CHECK: using BlockLoad = dpct::group::group_load<int, 4, dpct::group::group_load_algorithm::transpose>;
  using BlockLoad = cub::BlockLoad<int, 128, 4, cub::BLOCK_LOAD_TRANSPOSE>;
  __shared__ typename BlockLoad::TempStorage temp_storage;

  int thread_data[4] = {0};
  // CHECK: BlockLoad(temp_storage).load(item_ct1, d_data, thread_data, valid_items);
  BlockLoad(temp_storage).Load(d_data, thread_data, valid_items);
  printf("%3d: [%d, %d, %d, %d]\n", threadIdx.x, thread_data[0], thread_data[1], thread_data[2], thread_data[3]);
}

__global__ void SubGroupTransposeKernel(int *d_data, int valid_items) {
  // Specialize BlockLoad for a 1D block of 128 threads owning 4 integer items each
  // CHECK: using BlockLoad = dpct::group::group_load<int, 4, dpct::group::group_load_algorithm::sub_group_transpose>;
  using BlockLoad = cub::BlockLoad<int, 128, 4, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  __shared__ typename BlockLoad::TempStorage temp_storage;

  int thread_data[4] = {0};
  // CHECK: BlockLoad(temp_storage).load(item_ct1, d_data, thread_data, valid_items);
  BlockLoad(temp_storage).Load(d_data, thread_data, valid_items);
  printf("%3d: [%d, %d, %d, %d]\n", threadIdx.x, thread_data[0], thread_data[1], thread_data[2], thread_data[3]);
}

int main() {
  int *d_data;
  cudaMallocManaged(&d_data, sizeof(int) * 512);
  // cudaMemset(d_data, 0, sizeof(int) * 512);
  for (int i = 0; i < 512; ++i)
    d_data[i] = i;
  // CHECK: q_ct1.submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     sycl::stream stream_ct1(64 * 1024, 80, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     sycl::local_accessor<uint8_t, 1> temp_storage_acc(dpct::group::group_load<int, 4>::get_local_memory_size(sycl::range<3>(1, 1, 128).size()), cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         BlockedKernel(d_data, 5, item_ct1, stream_ct1, &temp_storage_acc[0]);
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  BlockedKernel<<<1, 128>>>(d_data, 5);
  cudaStreamSynchronize(0);

  // CHECK: q_ct1.submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     sycl::stream stream_ct1(64 * 1024, 80, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     sycl::local_accessor<uint8_t, 1> temp_storage_acc(dpct::group::group_load<int, 4>::get_local_memory_size(sycl::range<3>(1, 1, 128).size()), cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         StripedKernel(d_data, 128, item_ct1, stream_ct1, &temp_storage_acc[0]);
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  StripedKernel<<<1, 128>>>(d_data, 128);
  cudaStreamSynchronize(0);

// CHECK:  q_ct1.submit(
// CHECK:    [&](sycl::handler &cgh) {
// CHECK:      sycl::stream stream_ct1(64 * 1024, 80, cgh);
// CHECK:      sycl::local_accessor<uint8_t, 1> temp_storage_acc(dpct::group::group_load<int, 4, dpct::group::group_load_algorithm::transpose>::get_local_memory_size(sycl::range<3>(1, 1, 128).size()), cgh);
// CHECK:      cgh.parallel_for(
// CHECK:        sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
// CHECK:        [=](sycl::nd_item<3> item_ct1) {
// CHECK:          TransposeKernel(d_data, 128, item_ct1, stream_ct1, &temp_storage_acc[0]);
// CHECK:        });
// CHECK:    });
  TransposeKernel<<<1, 128>>>(d_data, 128);
  cudaStreamSynchronize(0);

// CHECK:  q_ct1.submit(
// CHECK:    [&](sycl::handler &cgh) {
// CHECK:      sycl::stream stream_ct1(64 * 1024, 80, cgh);
// CHECK:      sycl::local_accessor<uint8_t, 1> temp_storage_acc(dpct::group::group_load<int, 4, dpct::group::group_load_algorithm::sub_group_transpose>::get_local_memory_size(sycl::range<3>(1, 1, 128).size()), cgh);
// CHECK:      cgh.parallel_for(
// CHECK:        sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
// CHECK:        [=](sycl::nd_item<3> item_ct1) {
// CHECK:          SubGroupTransposeKernel(d_data, 128, item_ct1, stream_ct1, &temp_storage_acc[0]);
// CHECK:        });
// CHECK:    });
  SubGroupTransposeKernel<<<1, 128>>>(d_data, 128);
  cudaStreamSynchronize(0);

  return 0;
}
