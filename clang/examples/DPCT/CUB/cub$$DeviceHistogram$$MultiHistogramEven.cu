// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

void test(int num_pixels, unsigned char *d_samples, int *(&d_histogram)[3], int (&num_levels)[3], unsigned int (&lower_level)[3], unsigned int (&upper_level)[3], cudaStream_t S) {
  // Start
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::MultiHistogramEven<4, 3>(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_samples/*unsigned char **/, d_histogram/*int *(&)[3]*/, num_levels/*int(&)[3]*/, lower_level/*unsigned int(&)[3]*/, upper_level/*unsigned int(&)[3]*/, num_pixels/*int*/, S/*cudaStream_t*/);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceHistogram::MultiHistogramEven<4, 3>(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_samples/*unsigned char **/, d_histogram/*int *(&)[3]*/, num_levels/*int(&)[3]*/, lower_level/*unsigned int(&)[3]*/, upper_level/*unsigned int(&)[3]*/, num_pixels/*int*/, S/*cudaStream_t*/);
  // End
}
