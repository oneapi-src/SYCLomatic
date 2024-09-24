// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// UNSUPPORTED: system-windows
// RUN: dpct -in-root %S -out-root %T/blocklevel/blockstore %S/blockstore.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/blocklevel/blockstore/blockstore.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/blocklevel/blockstore/blockstore.dp.cpp -o %T/blocklevel/blockstore/blockstore.dp.o %}

#include <cub/cub.cuh>

__global__ void BlockedKernel(int *d_data, int valid_items) {
  // Specialize BlockStore for a 1D block of 128 threads owning 4 integer items each
  // CHECK: using BlockStore = dpct::group::group_store<int, 4>;
  using BlockStore = cub::BlockStore<int, 128, 4>;

  __shared__ typename BlockStore::TempStorage temp_storage;

  int thread_data[4];
  thread_data[0] = threadIdx.x * 4 + 0;
  thread_data[1] = threadIdx.x * 4 + 1;
  thread_data[2] = threadIdx.x * 4 + 2;
  thread_data[3] = threadIdx.x * 4 + 3;

  // CHECK: BlockStore(temp_storage).store(item_ct1, d_data, thread_data, valid_items);
  BlockStore(temp_storage).Store(d_data, thread_data, valid_items);
}

__global__ void StripedKernel(int *d_data, int valid_items) {
  // Specialize BlockStore for a 1D block of 128 threads owning 4 integer items each
  // CHECK: using BlockStore = dpct::group::group_store<int, 4, dpct::group::group_store_algorithm::striped>;
  using BlockStore = cub::BlockStore<int, 128, 4, cub::BLOCK_STORE_STRIPED>;

  __shared__ typename BlockStore::TempStorage temp_storage;

  int thread_data[4];
  thread_data[0] = threadIdx.x * 4 + 0;
  thread_data[1] = threadIdx.x * 4 + 1;
  thread_data[2] = threadIdx.x * 4 + 2;
  thread_data[3] = threadIdx.x * 4 + 3;
  // CHECK: BlockStore(temp_storage).store(item_ct1, d_data, thread_data, valid_items);
  BlockStore(temp_storage).Store(d_data, thread_data, valid_items);
}

int main() {
  int *d_data;
  cudaMallocManaged(&d_data, sizeof(int) * 512);
  cudaMemset(d_data, 0, sizeof(int) * 512);
  // CHECK: q_ct1.submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     sycl::local_accessor<uint8_t, 1> temp_storage_acc(dpct::group::group_store<int, 4>::get_local_memory_size(sycl::range<3>(1, 1, 128).size()), cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         BlockedKernel(d_data, 5, item_ct1, &temp_storage_acc[0]);
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  BlockedKernel<<<1, 128>>>(d_data, 5);
  cudaStreamSynchronize(0);
  for (int i = 0; i < 512; ++i)
    printf("%d%c", d_data[i], (i == 511 ? '\n' : ' '));
  cudaMemset(d_data, 0, sizeof(int) * 512);
  // CHECK: q_ct1.submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     sycl::local_accessor<uint8_t, 1> temp_storage_acc(dpct::group::group_store<int, 4>::get_local_memory_size(sycl::range<3>(1, 1, 128).size()), cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         StripedKernel(d_data, 5, item_ct1, &temp_storage_acc[0]);
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  StripedKernel<<<1, 128>>>(d_data, 5);
  cudaStreamSynchronize(0);
  for (int i = 0; i < 512; ++i)
    printf("%d%c", d_data[i], (i == 511 ? '\n' : ' '));
  return 0;
}
