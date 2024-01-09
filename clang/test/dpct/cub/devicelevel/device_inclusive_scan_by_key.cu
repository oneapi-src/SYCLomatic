// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/devicelevel/device_inclusive_scan_by_key %S/device_inclusive_scan_by_key.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/devicelevel/device_inclusive_scan_by_key/device_inclusive_scan_by_key.dp.cpp %s
// RUN: %if build_lit %{icpx -c -fsycl -DBUILD_TEST  %T/devicelevel/device_inclusive_scan_by_key/device_inclusive_scan_by_key.dp.cpp -o %T/devicelevel/device_inclusive_scan_by_key/device_inclusive_scan_by_key.dp.o %}
#ifndef  BUILD_TEST
// CHECK:#include <oneapi/dpl/execution>
// CHECK:#include <oneapi/dpl/algorithm>
#include <cub/cub.cuh>

int *d_key, *d_input, *d_output, num_items;
void *temp_storage;
size_t temp_storage_size;

struct CustomEqual {
  template <typename T>
  inline bool operator()(const T &lhs, const T &rhs) const {
    return lhs == rhs;
  }
} custom_eq;

void test1() {
  cudaStream_t stream/*=undefined value*/;

  // CHECK: oneapi::dpl::inclusive_scan_by_key(oneapi::dpl::execution::device_policy(q_ct1), d_key, d_key + num_items, d_input, d_output, 0, std::equal_to<>(), sycl::maximum<>());
  cub::DeviceScan::InclusiveScanByKey(temp_storage, temp_storage_size, d_key, d_input,d_output, cub::Max(), num_items);

  // CHECK: oneapi::dpl::inclusive_scan_by_key(oneapi::dpl::execution::device_policy(q_ct1), d_key, d_key + num_items, d_input, d_output, 0, std::equal_to<>(), sycl::maximum<>());
  cub::DeviceScan::InclusiveScanByKey(temp_storage, temp_storage_size, d_key, d_input,d_output, cub::Max(), num_items, custom_eq);

  // CHECK: oneapi::dpl::inclusive_scan_by_key(oneapi::dpl::execution::device_policy(q_ct1), d_key, d_key + num_items, d_input, d_output, 0, std::equal_to<>(), sycl::maximum<>());
  cub::DeviceScan::InclusiveScanByKey(temp_storage, temp_storage_size, d_key, d_input,d_output, cub::Max(), num_items, cub::Equality());

  // CHECK: oneapi::dpl::inclusive_scan_by_key(oneapi::dpl::execution::device_policy(*stream), d_key, d_key + num_items, d_input, d_output, 0, custom_eq, sycl::maximum<>());
  cub::DeviceScan::InclusiveScanByKey(temp_storage, temp_storage_size, d_key, d_input,d_output, cub::Max(), num_items, custom_eq, stream);

  // CHECK: oneapi::dpl::inclusive_scan_by_key(oneapi::dpl::execution::device_policy(*stream), d_key, d_key + num_items, d_input, d_output, 0, std::equal_to<>(), sycl::maximum<>());
  cub::DeviceScan::InclusiveScanByKey(temp_storage, temp_storage_size, d_key, d_input,d_output, cub::Max(), num_items, cub::Equality(), stream);
}

void test2() {
  // CHECK: DPCT1026:{{.*}}
  // CHECK: oneapi::dpl::inclusive_scan_by_key(oneapi::dpl::execution::device_policy(q_ct1), d_key, d_key + num_items, d_input, d_output, 0, std::equal_to<>(), sycl::maximum<>());
  void *temp_storage = nullptr;
  size_t temp_storage_size = 0;
  cub::DeviceScan::InclusiveScanByKey(temp_storage, temp_storage_size, d_key, d_input,d_output, cub::Max(), 0, num_items);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceScan::InclusiveScanByKey(temp_storage, temp_storage_size, d_key, d_input,d_output, cub::Max(), 0, num_items);
  cudaFree(temp_storage);
}

void test3() {
  // CHECK: DPCT1026:{{.*}}
  // CHECK: oneapi::dpl::inclusive_scan_by_key(oneapi::dpl::execution::device_policy(q_ct1), d_key, d_key + num_items, d_input, d_output, 0, std::equal_to<>(), sycl::maximum<>());
  void *temp_storage = nullptr;
  size_t temp_storage_size = 0;
  cub::DeviceScan::InclusiveScanByKey(temp_storage, temp_storage_size, d_key, d_input,d_output, cub::Max(), num_items, custom_eq);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceScan::InclusiveScanByKey(temp_storage, temp_storage_size, d_key, d_input,d_output, cub::Max(), num_items, custom_eq);
  cudaFree(temp_storage);
}

void test4() {
  // CHECK: DPCT1026:{{.*}}
  // CHECK: oneapi::dpl::inclusive_scan_by_key(oneapi::dpl::execution::device_policy(q_ct1), d_key, d_key + num_items, d_input, d_output, 0, std::equal_to<>(), sycl::maximum<>());
  void *temp_storage = nullptr;
  size_t temp_storage_size = 0;
  cub::DeviceScan::InclusiveScanByKey(temp_storage, temp_storage_size, d_key, d_input,d_output, cub::Max(), num_items, cub::Equality());
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceScan::InclusiveScanByKey(temp_storage, temp_storage_size, d_key, d_input,d_output, cub::Max(), num_items, cub::Equality());
  cudaFree(temp_storage);
}

void test5() {
  // CHECK: DPCT1026:{{.*}}
  // CHECK: oneapi::dpl::inclusive_scan_by_key(oneapi::dpl::execution::device_policy(*stream), d_key, d_key + num_items, d_input, d_output, 0, custom_eq, sycl::maximum<>());
  cudaStream_t stream/*=undefined value*/;
  void *temp_storage = nullptr;
  size_t temp_storage_size = 0;
  cub::DeviceScan::InclusiveScanByKey(temp_storage, temp_storage_size, d_key, d_input,d_output, cub::Max(), num_items, custom_eq, stream);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceScan::InclusiveScanByKey(temp_storage, temp_storage_size, d_key, d_input,d_output, cub::Max(), num_items, custom_eq, stream);
  cudaFree(temp_storage);
}

void test6() {
  // CHECK: DPCT1026:{{.*}}
  // CHECK: oneapi::dpl::inclusive_scan_by_key(oneapi::dpl::execution::device_policy(*stream), d_key, d_key + num_items, d_input, d_output, 0, std::equal_to<>(), sycl::maximum<>());
  cudaStream_t stream/*=undefined value*/;
  void *temp_storage = nullptr;
  size_t temp_storage_size = 0;
  cub::DeviceScan::InclusiveScanByKey(temp_storage, temp_storage_size, d_key, d_input,d_output, cub::Max(), num_items, cub::Equality(), stream);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceScan::InclusiveScanByKey(temp_storage, temp_storage_size, d_key, d_input,d_output, cub::Max(), num_items, cub::Equality(), stream);
  cudaFree(temp_storage);
}
#endif
