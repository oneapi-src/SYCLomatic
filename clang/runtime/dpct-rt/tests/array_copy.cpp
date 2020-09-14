// RUN: dpcpp array_copy.cpp -o array_copy

#include <stdio.h>

#include "../include/dpct.hpp"

bool check(const int *in, const int *out, size_t size) {
  for (auto i = 0; i < size; ++i) {
    if (in[i] != out[i])
      return false;
  }
  return true;
}

bool check_zero(const int *in, size_t size) {
  for (auto i = 0; i < size; ++i) {
    if (in[i])
      return false;
  }
  return true;
}

void init(int *in, size_t size) {
  for (auto i = 0; i < size; ++i) {
    in[i] = i;
  }
}

void init_zero(int *in, size_t size) {
  for (auto i = 0; i < size; ++i) {
    in[i] = 0;
  }
}

void print(int *in, size_t size) {
  for (auto i = 0; i < size; ++i)
    printf("[%d]: %d\n", i, in[i]);
  printf("\n");
}

int main() {
  dpct::image_matrix_p ain, aout;
  dpct::image_channel chn =
      dpct::create_image_channel(32, 0, 0, 0, dpct::image_channel_data_type::signed_int);
  int *in, *out;
  sycl::range<2> size = sycl::range<2>(256, 1);

  in = (int *)std::malloc(size.size() * sizeof(int));
  out = (int *)std::malloc(size.size() * sizeof(int));
  ain = new dpct::image_matrix(chn, size);
  aout = new dpct::image_matrix(chn, size);

  init(in, size.size());
  init_zero(out, size.size());
  /// ain init zero.
  dpct::dpct_memcpy(ain->to_pitched_data(), sycl::id<3>(0, 0, 0), dpct::pitched_data(out, 256 * sizeof(int), 256 * sizeof(int), 1), sycl::id<3>(0, 0, 0), sycl::range<3>(256 * sizeof(int), 1, 1));
  /// aout init zero.
  dpct::dpct_memcpy(aout->to_pitched_data(), sycl::id<3>(0, 0, 0), dpct::pitched_data(out, 256 * sizeof(int), 256 * sizeof(int), 1), sycl::id<3>(0, 0, 0), sycl::range<3>(256 * sizeof(int), 1, 1));
  
  /// copy in [0, 224) to ain [32, 256).
  dpct::dpct_memcpy(ain->to_pitched_data(), sycl::id<3>(32 * sizeof(int), 0, 0), dpct::pitched_data(in, 256 * sizeof(int), 256 * sizeof(int), 1), sycl::id<3>(0, 0, 0), sycl::range<3>(224 * sizeof(int), 1, 1));
  /// copy ain [64, 224) to aout [32, 192).
  dpct::dpct_memcpy(aout->to_pitched_data(), sycl::id<3>(32 * sizeof(int), 0, 0), ain->to_pitched_data(), sycl::id<3>(64 * sizeof(int), 0, 0), sycl::range<3>(160 * sizeof(int), 1, 1));
  /// copy aout [0, 224) to out [0, 224).
  dpct::dpct_memcpy(dpct::pitched_data(out, 256 * sizeof(int), 256 * sizeof(int), 1), sycl::id<3>(0, 0, 0), aout->to_pitched_data(), sycl::id<3>(0, 0, 0), sycl::range<3>(224 * sizeof(int), 1, 1));

  if (check(in + 32, out + 32, 128) && check_zero(out, 32) && check_zero(out + 192, 64))
    printf("Success!\n");
  else
    printf("Fail!\n");
}