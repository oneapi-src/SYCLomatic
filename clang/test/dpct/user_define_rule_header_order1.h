#include <cub/cub.cuh>
#include <stddef.h>

int n, *d_in, *d_out;
void *tmp;
size_t tmp_size;

#define CUB_WRAPPER(func, ...) do {                                       \
  void *temp_storage = nullptr;                                           \
  size_t temp_storage_bytes = 0;                                          \
  func(temp_storage, temp_storage_bytes, __VA_ARGS__);                    \
} while (false)

inline void test1() {
  CUB_WRAPPER(cub::DeviceScan::InclusiveSum, d_in, d_out, n);
}
