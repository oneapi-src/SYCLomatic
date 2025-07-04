#include <thrust/extrema.h>

void min_test() {
  // clang-format off
  // Start
  struct key_value {
    int key;
    int value;
  };
  struct compare_key_value {
    __host__ __device__ bool operator()(key_value lhs, key_value rhs) {
      return lhs.key < rhs.key;
    }
  };
  key_value a = {13, 0};
  key_value b = {7, 1};
  key_value smaller = thrust::min(a, b, compare_key_value());
  int value = thrust::min(1, 2);
  // End
  // clang-format on
}
