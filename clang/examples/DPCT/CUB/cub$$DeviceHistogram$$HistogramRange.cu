// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

void test(int num_samples, float *d_samples, int *d_histogram, int num_levels, float *d_levels) {
  // Start
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::HistogramRange(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_samples/*float **/, d_histogram/*int **/, num_levels/*int*/, d_levels/*float **/, num_samples/*int*/);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceHistogram::HistogramRange(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_samples/*float **/, d_histogram/*int **/, num_levels/*int*/, d_levels/*float **/, num_samples/*int*/);
  // End
}
