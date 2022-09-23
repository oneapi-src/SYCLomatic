// RUN: echo
#include <cuda.h>
#include <cuda_runtime_api.h>

template <typename acc_t, typename func_t>
struct func_wrapper_t {
  func_t combine;
  // CHECK: func_wrapper_t(const func_t &op) : combine(op) {}
  __device__ func_wrapper_t(const func_t &op) : combine(op) {}
  // CHECK: acc_t reduce(acc_t x, acc_t y) const { return combine(x, y); }
  __device__ acc_t reduce(acc_t x, acc_t y) const { return combine(x, y); }
};

template <typename acc_t, typename func_t>
// CHECK: func_wrapper_t<acc_t, func_t> func_wrapper(const func_t &op) {
__device__ func_wrapper_t<acc_t, func_t> func_wrapper(const func_t &op) {
  return func_wrapper_t<acc_t, func_t>{op};
}