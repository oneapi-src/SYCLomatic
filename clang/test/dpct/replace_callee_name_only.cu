// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.2, cuda-11.4
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.2, v11.4
// RUN: cat %s > %T/replace_callee_name_only.cu
// RUN: cat %S/replace_callee_name_only.yaml > %T/replace_callee_name_only.yaml
// RUN: cd %T
// RUN: rm -rf %T/replace_callee_name_only_output
// RUN: mkdir %T/replace_callee_name_only_output
// RUN: dpct --format-range=none -out-root %T/replace_callee_name_only_output replace_callee_name_only.cu --cuda-include-path="%cuda-path/include" --usm-level=none --rule-file=replace_callee_name_only.yaml -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/replace_callee_name_only_output/replace_callee_name_only.dp.cpp --match-full-lines replace_callee_name_only.cu

#include <cub/cub.cuh>
#include <stddef.h>

struct CustomMin {
  template <typename OutputT>
  __host__ __device__ __forceinline__ OutputT
  operator()(const OutputT& a, const OutputT& b) const {
    if (std::isnan(a)) {
      return a;
    } else if (std::isnan(b)) {
      return b;
    }
    return std::min<OutputT>(a, b);
  }
} min_op;

int n, initial_value, *d_in, *d_out, *d_offsets;
void *tmp;
size_t tmp_size;

#define CUB_WRAPPER(func, ...) do {                                       \
  void *temp_storage = nullptr;                                           \
  size_t temp_storage_bytes = 0;                                          \
  func(temp_storage, temp_storage_bytes, __VA_ARGS__);                    \
} while (false)

void dummy_func(int *, int *, int) {}
void dummy_func(void *, size_t, int *, int *, int) {}
void dummy_func(void *, size_t, int *, int *, int, int *, int *, CustomMin, int, cudaStream_t) {}
cudaStream_t getCudaStream() { return nullptr; }
cudaStream_t getXpuStream() { return nullptr; }

void test0() {
  // CHECK: CUB_WRAPPER(dummy_func, d_in, d_out, n);
  CUB_WRAPPER(cub::DeviceReduce::Sum, d_in, d_out, n);
}

void test1() {
  // CHECK: dummy_func(d_in, d_out, n);
  cub::DeviceScan::ExclusiveSum(tmp, tmp_size, d_in, d_out, n);
}

void test2() {
  // CHECK: dummy_func(tmp, tmp_size, d_in, d_out, n, d_offsets, d_offsets, min_op, initial_value, getXpuStream());
  cub::DeviceSegmentedReduce::Reduce(tmp, tmp_size, d_in, d_out, n, d_offsets, d_offsets, min_op, initial_value, getCudaStream());
  return c;
}
