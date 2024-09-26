// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct -format-range=none -in-root %S -out-root %T/blocklevel/blockexchange %S/blockexchange.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/blocklevel/blockexchange/blockexchange.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/blocklevel/blockexchange/blockexchange.dp.cpp -o %T/blocklevel/blockexchange/blockexchange.dp.o %}

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <iostream>

__global__ void StripedToBlockedKernel(int *d_data) {
  // CHECK: typedef dpct::group::exchange<int, 4> BlockExchange;
  // CHECK-NEXT: using BlockLoad = dpct::group::group_load<int, 4, dpct::group::group_load_algorithm::striped>;
  // CHECK-NEXT: using BlockStore = dpct::group::group_store<int, 4, dpct::group::group_store_algorithm::striped>;
  // CHECK-NOT: __shared__ typename BlockLoad::TempStorage temp_storage_load;
  // CHECK-NOT: __shared__ typename BlockStore::TempStorage temp_storage_store;
  // CHECK-NOT: __shared__ typename BlockExchange::TempStorage temp_storage;
  typedef cub::BlockExchange<int, 128, 4> BlockExchange;
  using BlockLoad = cub::BlockLoad<int, 128, 4, cub::BLOCK_LOAD_STRIPED>;
  using BlockStore = cub::BlockStore<int, 128, 4, cub::BLOCK_STORE_STRIPED>;
  __shared__ typename BlockLoad::TempStorage temp_storage_load;
  __shared__ typename BlockStore::TempStorage temp_storage_store;
  __shared__ typename BlockExchange::TempStorage temp_storage;
  int thread_data[4];
  // CHECK: BlockLoad(temp_storage_load).load(item_ct1, d_data, thread_data);
  // CHECK-NEXT: BlockExchange(temp_storage).striped_to_blocked(item_ct1, thread_data, thread_data);
  // CHECK-NEXT: BlockStore(temp_storage_store).store(item_ct1, d_data, thread_data);
  BlockLoad(temp_storage_load).Load(d_data, thread_data);
  BlockExchange(temp_storage).StripedToBlocked(thread_data, thread_data);
  BlockStore(temp_storage_store).Store(d_data, thread_data);
}

__global__ void BlockedToStripedKernel(int *d_data) {
  // CHECK: typedef dpct::group::exchange<int, 4> BlockExchange;
  // CHECK-NEXT: using BlockLoad = dpct::group::group_load<int, 4, dpct::group::group_load_algorithm::striped>;
  // CHECK-NEXT: using BlockStore = dpct::group::group_store<int, 4, dpct::group::group_store_algorithm::striped>;
  // CHECK-NOT: __shared__ typename BlockLoad::TempStorage temp_storage_load;
  // CHECK-NOT: __shared__ typename BlockStore::TempStorage temp_storage_store;
  // CHECK-NOT: __shared__ typename BlockExchange::TempStorage temp_storage;
  typedef cub::BlockExchange<int, 128, 4> BlockExchange;
  using BlockLoad = cub::BlockLoad<int, 128, 4, cub::BLOCK_LOAD_STRIPED>;
  using BlockStore = cub::BlockStore<int, 128, 4, cub::BLOCK_STORE_STRIPED>;
  __shared__ typename BlockLoad::TempStorage temp_storage_load;
  __shared__ typename BlockStore::TempStorage temp_storage_store;
  __shared__ typename BlockExchange::TempStorage temp_storage;
  int thread_data[4];
  // CHECK: BlockLoad(temp_storage_load).load(item_ct1, d_data, thread_data);
  // CHECK-NEXT: BlockExchange(temp_storage).blocked_to_striped(item_ct1, thread_data, thread_data);
  // CHECK-NEXT: BlockStore(temp_storage_store).store(item_ct1, d_data, thread_data);
  BlockLoad(temp_storage_load).Load(d_data, thread_data);
  BlockExchange(temp_storage).BlockedToStriped(thread_data, thread_data);
  BlockStore(temp_storage_store).Store(d_data, thread_data);
}

__global__ void ScatterToBlockedKernel(int *d_data, int *d_rank) {
  // CHECK: using BlockExchange = dpct::group::exchange<int, 4>;
  // CHECK-NOT:__shared__ typename BlockExchange::TempStorage temp_storage;
  using BlockExchange = cub::BlockExchange<int, 128, 4>;
  __shared__ typename BlockExchange::TempStorage temp_storage;
  int thread_data[4], thread_rank[4];
  // CHECK: dpct::group::load_direct_striped(item_ct1, d_data, thread_data);
  // CHECK-NEXT: dpct::group::load_direct_striped(item_ct1, d_rank, thread_rank);
  // CHECK-NEXT: BlockExchange(temp_storage).scatter_to_blocked(item_ct1, thread_data, thread_rank);
  // CHECK-NEXT: dpct::group::store_direct_striped(item_ct1, d_data, thread_data);
  cub::LoadDirectStriped<128>(threadIdx.x, d_data, thread_data);
  cub::LoadDirectStriped<128>(threadIdx.x, d_rank, thread_rank);
  BlockExchange(temp_storage).ScatterToBlocked(thread_data, thread_rank);
  cub::StoreDirectStriped<128>(threadIdx.x, d_data, thread_data);
}

__global__ void ScatterToStripedKernel(int *d_data, int *d_rank) {
  // CHECK: using BlockExchange = dpct::group::exchange<int, 4>;
  // CHECK-NOT:__shared__ typename BlockExchange::TempStorage temp_storage;
  using BlockExchange = cub::BlockExchange<int, 128, 4>;
  __shared__ typename BlockExchange::TempStorage temp_storage;
  int thread_data[4], thread_rank[4];
  // CHECK: dpct::group::load_direct_striped(item_ct1, d_data, thread_data);
  // CHECK-NEXT: dpct::group::load_direct_striped(item_ct1, d_rank, thread_rank);
  // CHECK-NEXT: BlockExchange(temp_storage).scatter_to_striped(item_ct1, thread_data, thread_rank);
  // CHECK-NEXT: dpct::group::store_direct_striped(item_ct1, d_data, thread_data);
  cub::LoadDirectStriped<128>(threadIdx.x, d_data, thread_data);
  cub::LoadDirectStriped<128>(threadIdx.x, d_rank, thread_rank);
  BlockExchange(temp_storage).ScatterToStriped(thread_data, thread_rank);
  cub::StoreDirectStriped<128>(threadIdx.x, d_data, thread_data);
}

bool test_striped_to_blocked() {
  int *d_data;
  cudaMallocManaged(&d_data, sizeof(int) * 512);
  for (int i = 0; i < 128; i++) {
    d_data[4 * i + 0] = i;
    d_data[4 * i + 1] = i + 1 * 128;
    d_data[4 * i + 2] = i + 2 * 128;
    d_data[4 * i + 3] = i + 3 * 128;
  }

  // CHECK: q_ct1.submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     sycl::local_accessor<uint8_t, 1> temp_storage_load_acc(dpct::group::group_load<int, 4>::get_local_memory_size(sycl::range<3>(1, 1, 128).size()), cgh);
  // CHECK-NEXT:     sycl::local_accessor<uint8_t, 1> temp_storage_store_acc(dpct::group::group_store<int, 4>::get_local_memory_size(sycl::range<3>(1, 1, 128).size()), cgh);
  // CHECK-NEXT:     sycl::local_accessor<uint8_t, 1> temp_storage_acc(dpct::group::exchange<int, 4>::get_local_memory_size(sycl::range<3>(1, 1, 128).size()), cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         StripedToBlockedKernel(d_data, item_ct1, &temp_storage_load_acc[0], &temp_storage_store_acc[0], &temp_storage_acc[0]);
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  StripedToBlockedKernel<<<1, 128>>>(d_data);
  cudaDeviceSynchronize();

  for (int i = 0; i < 512; ++i) {
    if (d_data[i] != i) {
      std::cout << "test_striped_to_blocked failed\n";
      std::ostream_iterator<int> Iter(std::cout, ", ");
      std::copy(d_data, d_data + 512, Iter);
      std::cout << std::endl;
      return false;
    }
  }

  std::cout << "test_striped_to_blocked pass\n";
  return true;
}

bool test_blocked_to_striped() {
  int *d_data, expected[512];
  cudaMallocManaged(&d_data, sizeof(int) * 512);
  for (int i = 0; i < 512; ++i)
    d_data[i] = i;

  // CHECK: q_ct1.submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     sycl::local_accessor<uint8_t, 1> temp_storage_load_acc(dpct::group::group_load<int, 4>::get_local_memory_size(sycl::range<3>(1, 1, 128).size()), cgh);
  // CHECK-NEXT:     sycl::local_accessor<uint8_t, 1> temp_storage_store_acc(dpct::group::group_store<int, 4>::get_local_memory_size(sycl::range<3>(1, 1, 128).size()), cgh);
  // CHECK-NEXT:     sycl::local_accessor<uint8_t, 1> temp_storage_acc(dpct::group::exchange<int, 4>::get_local_memory_size(sycl::range<3>(1, 1, 128).size()), cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         BlockedToStripedKernel(d_data, item_ct1, &temp_storage_load_acc[0], &temp_storage_store_acc[0], &temp_storage_acc[0]);
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  BlockedToStripedKernel<<<1, 128>>>(d_data);
  cudaDeviceSynchronize();

  for (int i = 0; i < 128; i++) {
    expected[4 * i + 0] = i;
    expected[4 * i + 1] = i + 1 * 128;
    expected[4 * i + 2] = i + 2 * 128;
    expected[4 * i + 3] = i + 3 * 128;
  }

  for (int i = 0; i < 512; ++i) {
    if (expected[i] != d_data[i]) {
      std::cout << "test_blocked_to_striped failed\n";
      std::ostream_iterator<int> Iter(std::cout, ", ");
      std::copy(d_data, d_data + 512, Iter);
      std::cout << std::endl;
      return false;
    }
  }
  std::cout << "test_blocked_to_striped pass\n";
  return true;
}

bool test_scatter_to_blocked() {
  int *d_data, *d_rank;
  cudaMallocManaged(&d_data, sizeof(int) * 512);
  cudaMallocManaged(&d_rank, sizeof(int) * 512);
  for (int i = 0; i < 128; i++) {
    d_data[4 * i + 0] = i;
    d_data[4 * i + 1] = i + 1 * 128;
    d_data[4 * i + 2] = i + 2 * 128;
    d_data[4 * i + 3] = i + 3 * 128;
    d_rank[4 * i + 0] = i * 4 + 0;
    d_rank[4 * i + 1] = i * 4 + 1;
    d_rank[4 * i + 2] = i * 4 + 2;
    d_rank[4 * i + 3] = i * 4 + 3;
  }

  // CHECK: q_ct1.submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     sycl::local_accessor<uint8_t, 1> temp_storage_acc(dpct::group::exchange<int, 4>::get_local_memory_size(sycl::range<3>(1, 1, 128).size()), cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         ScatterToBlockedKernel(d_data, d_rank, item_ct1, &temp_storage_acc[0]);
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  ScatterToBlockedKernel<<<1, 128>>>(d_data, d_rank);
  cudaDeviceSynchronize();

  for (int i = 0; i < 512; ++i) {
    if (d_data[i] != i) {
      std::cout << "test_scatter_to_blocked failed\n";
      std::ostream_iterator<int> Iter(std::cout, ", ");
      std::copy(d_data, d_data + 512, Iter);
      std::cout << std::endl;
      return false;
    }
  }

  std::cout << "test_scatter_to_blocked pass\n";
  return true;
}

bool test_scatter_to_striped() {
  int *d_data, *d_rank, expected[512];
  cudaMallocManaged(&d_data, sizeof(int) * 512);
  cudaMallocManaged(&d_rank, sizeof(int) * 512);
  for (int i = 0; i < 512; ++i)
    d_data[i] = i;

  d_rank[0] = 0;
  d_rank[128] = 1;
  d_rank[256] = 2;
  d_rank[384] = 3;
  for (int i = 1; i < 128; i++) {
    d_rank[0 * 128 + i] = d_rank[0 * 128 + i - 1] + 4;
    d_rank[1 * 128 + i] = d_rank[1 * 128 + i - 1] + 4;
    d_rank[2 * 128 + i] = d_rank[2 * 128 + i - 1] + 4;
    d_rank[3 * 128 + i] = d_rank[3 * 128 + i - 1] + 4;
  }

  // CHECK: q_ct1.submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     sycl::local_accessor<uint8_t, 1> temp_storage_acc(dpct::group::exchange<int, 4>::get_local_memory_size(sycl::range<3>(1, 1, 128).size()), cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         ScatterToStripedKernel(d_data, d_rank, item_ct1, &temp_storage_acc[0]);
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  ScatterToStripedKernel<<<1, 128>>>(d_data, d_rank);
  cudaDeviceSynchronize();

  for (int i = 0; i < 128; i++) {
    expected[4 * i + 0] = i + 0 * 128;
    expected[4 * i + 1] = i + 1 * 128;
    expected[4 * i + 2] = i + 2 * 128;
    expected[4 * i + 3] = i + 3 * 128;
  }

  for (int i = 0; i < 512; ++i) {
    if (expected[i] != d_data[i]) {
      std::cout << "test_blocked_to_striped failed\n";
      std::ostream_iterator<int> Iter(std::cout, ", ");
      std::copy(d_data, d_data + 512, Iter);
      std::cout << std::endl;
      return false;
    }
  }
  std::cout << "test_blocked_to_striped pass\n";
  return true;
}

int main() {
  return !(test_blocked_to_striped() && test_striped_to_blocked() &&
           test_scatter_to_blocked() && test_scatter_to_striped());
}
