// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/devicelevel/device_histgram %S/device_histgram.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/devicelevel/device_histgram/device_histgram.dp.cpp --match-full-lines %s

// CHECK:#include <dpct/dpl_utils.hpp>
#include <cub/cub.cuh>
#include <cstddef>

// clang-format off
bool histgram_even() {
  int num_samples;
  float *d_samples;
  int *d_histogram;
  int num_levels;
  float lower_level;
  float upper_level;

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels, lower_level, upper_level, num_samples);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // CHECK: DPCT1026:{{.*}}: The call to cub::DeviceHistogram::HistogramEven was removed because this call is redundant in SYCL.
  // CHECK: dpct::histogram_even(oneapi::dpl::execution::device_policy(q_ct1), d_samples, d_histogram, num_levels, lower_level, upper_level, num_samples);
  cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels, lower_level, upper_level, num_samples);
  return true;
}

bool histgram_even_roi() {
  int num_row_samples;
  int num_rows;
  size_t row_stride_bytes = 7 * sizeof(float);
  float *d_samples;
  int *d_histogram;
  int num_levels;
  float lower_level;
  float upper_level = 12.0;

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels, lower_level, upper_level, num_row_samples, num_rows, row_stride_bytes);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // CHECK: DPCT1026:{{.*}}: The call to cub::DeviceHistogram::HistogramEven was removed because this call is redundant in SYCL.
  // CHECK: dpct::histogram_even_roi(oneapi::dpl::execution::device_policy(q_ct1), d_samples, d_histogram, num_levels, lower_level, upper_level, num_row_samples, num_rows, row_stride_bytes);
  cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels, lower_level, upper_level, num_row_samples, num_rows, row_stride_bytes);
  return true;
}

bool multi_histgram_even() {
  int num_pixels = 5;
  unsigned char *d_samples;
  int *d_histogram[3];
  int num_levels[3] = {257, 257, 257};
  unsigned int lower_level[3] = {0, 0, 0};
  unsigned int upper_level[3] = {256, 256, 256};

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::MultiHistogramEven<4, 3>(d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels, lower_level, upper_level, num_pixels);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // CHECK: DPCT1026:{{.*}}: The call to cub::DeviceHistogram::MultiHistogramEven was removed because this call is redundant in SYCL.
  // CHECK: dpct::multi_histogram_even<4, 3>(oneapi::dpl::execution::device_policy(q_ct1), d_samples, d_histogram, num_levels, lower_level, upper_level, num_pixels);
  cub::DeviceHistogram::MultiHistogramEven<4, 3>(d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels, lower_level, upper_level, num_pixels);

  return true;
}

bool multi_histgram_even_roi() {
  int num_row_pixels = 3;
  int num_rows = 2;
  size_t row_stride_bytes;
  unsigned char *d_samples;
  int *d_histogram[3];
  int num_levels[3] = {257, 257, 257};
  unsigned int lower_level[3] = {0, 0, 0};
  unsigned int upper_level[3] = {256, 256, 256};
  
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::MultiHistogramEven<4, 3>(d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels, lower_level, upper_level, num_row_pixels, num_rows, row_stride_bytes);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // CHECK: DPCT1026:{{.*}}: The call to cub::DeviceHistogram::MultiHistogramEven was removed because this call is redundant in SYCL.
  // CHECK: dpct::multi_histogram_even_roi<4, 3>(oneapi::dpl::execution::device_policy(q_ct1), d_samples, d_histogram, num_levels, lower_level, upper_level, num_row_pixels, num_rows, row_stride_bytes);
  cub::DeviceHistogram::MultiHistogramEven<4, 3>(d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels, lower_level, upper_level, num_row_pixels, num_rows, row_stride_bytes);

  return true;
}

bool histgram_range() {
  int num_samples = 10;
  float *d_samples;
  int *d_histogram;
  int num_levels = 7; // (seven level boundaries for six bins)
  float *d_levels;

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::HistogramRange(d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels, d_levels, num_samples);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // CHECK: DPCT1026:{{.*}}: The call to cub::DeviceHistogram::HistogramRange was removed because this call is redundant in SYCL.
  // CHECK: dpct::histogram_range(oneapi::dpl::execution::device_policy(q_ct1), d_samples, d_histogram, num_levels, d_levels, num_samples);
  cub::DeviceHistogram::HistogramRange(d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels, d_levels, num_samples);

  return true;
}

bool histgram_range_roi() {
  int num_row_samples = 5;
  int num_rows = 2;
  int row_stride_bytes = 7 * sizeof(float);
  float *d_samples;
  int *d_histogram;
  int num_levels = 7; // (seven level boundaries for six bins)
  float *d_levels;
  
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::HistogramRange(d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels,d_levels, num_row_samples, num_rows, row_stride_bytes);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // CHECK: DPCT1026:{{.*}}: The call to cub::DeviceHistogram::HistogramRange was removed because this call is redundant in SYCL.
  // CHECK: dpct::histogram_range_roi(oneapi::dpl::execution::device_policy(q_ct1), d_samples, d_histogram, num_levels, d_levels, num_row_samples, num_rows, row_stride_bytes);
  cub::DeviceHistogram::HistogramRange(d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels,d_levels, num_row_samples, num_rows, row_stride_bytes);


  return true;
}

bool multi_histgram_range() {
  int num_pixels = 5;
  unsigned char *d_samples;
  unsigned int *d_histogram[3];
  int num_levels[3] = {5, 5, 5};
  unsigned int *d_levels[3];
  
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::MultiHistogramRange<4, 3>(d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels,d_levels, num_pixels);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // CHECK: DPCT1026:{{.*}}: The call to cub::DeviceHistogram::MultiHistogramRange was removed because this call is redundant in SYCL.
  // CHECK: dpct::multi_histogram_range<4, 3>(oneapi::dpl::execution::device_policy(q_ct1), d_samples, d_histogram, num_levels, d_levels, num_pixels);
  cub::DeviceHistogram::MultiHistogramRange<4, 3>(d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels, d_levels, num_pixels);

  return true;
}

bool multi_histgram_range_roi() {
  int num_row_pixels = 3;
  int num_rows = 2;
  size_t row_stride_bytes;
  // clang-format off
  unsigned char *d_samples;
  // clang-format on
  int *d_histogram[3];
  int num_levels[3] = {5, 5, 5};
  unsigned int *d_levels[3];

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::MultiHistogramRange<4, 3>(d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels, d_levels, num_row_pixels, num_rows, row_stride_bytes);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // CHECK: DPCT1026:{{.*}}: The call to cub::DeviceHistogram::MultiHistogramRange was removed because this call is redundant in SYCL.
  // CHECK: dpct::multi_histogram_range_roi<4, 3>(oneapi::dpl::execution::device_policy(q_ct1), d_samples, d_histogram, num_levels, d_levels, num_row_pixels, num_rows, row_stride_bytes);
  cub::DeviceHistogram::MultiHistogramRange<4, 3>(d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels, d_levels, num_row_pixels, num_rows, row_stride_bytes);

  return true;
}

bool histgram_even_stream() {
  int num_samples = 10;
  float *d_samples;
  int *d_histogram;
  int num_levels = 7;       // (seven level boundaries for six bins)
  float lower_level = 0.0; 
  float upper_level = 12.0;

  cudaStream_t S;
  cudaStreamCreate(&S);

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels, lower_level, upper_level, num_samples, S);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // CHECK: DPCT1026:{{.*}}: The call to cub::DeviceHistogram::HistogramEven was removed because this call is redundant in SYCL.
  // CHECK: dpct::histogram_even(oneapi::dpl::execution::device_policy(*S), d_samples, d_histogram, num_levels, lower_level, upper_level, num_samples);
  cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels, lower_level, upper_level, num_samples, S);

  return true;
}

bool histgram_even_roi_stream() {
  int num_row_samples = 5;
  int num_rows = 2;
  size_t row_stride_bytes = 7 * sizeof(float);
  float *d_samples;
  int *d_histogram;
  int num_levels = 7;       // (seven level boundaries for six bins)
  float lower_level = 0.0; 
  float upper_level = 12.0;

  cudaStream_t S;
  cudaStreamCreate(&S);
  
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels, lower_level, upper_level, num_row_samples, num_rows, row_stride_bytes, S);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // CHECK: DPCT1026:{{.*}}: The call to cub::DeviceHistogram::HistogramEven was removed because this call is redundant in SYCL.
  // CHECK: dpct::histogram_even_roi(oneapi::dpl::execution::device_policy(*S), d_samples, d_histogram, num_levels, lower_level, upper_level, num_row_samples, num_rows, row_stride_bytes);
  cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels, lower_level, upper_level, num_row_samples, num_rows, row_stride_bytes, S);

  return true;
}

bool multi_histgram_even_stream() {
  int num_pixels = 5;
  unsigned char *d_samples;
  int *d_histogram[3];
  int num_levels[3] = {257, 257, 257};
  unsigned int lower_level[3] = {0, 0, 0};
  unsigned int upper_level[3] = {256, 256, 256};

  cudaStream_t S;
  cudaStreamCreate(&S);

  
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::MultiHistogramEven<4, 3>(d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels, lower_level, upper_level, num_pixels, S);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // CHECK: DPCT1026:{{.*}}: The call to cub::DeviceHistogram::MultiHistogramEven was removed because this call is redundant in SYCL.
  // CHECK: dpct::multi_histogram_even<4, 3>(oneapi::dpl::execution::device_policy(*S), d_samples, d_histogram, num_levels, lower_level, upper_level, num_pixels);
  cub::DeviceHistogram::MultiHistogramEven<4, 3>(d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels, lower_level, upper_level, num_pixels, S);

  return true;
}

bool multi_histgram_even_roi_stream() {
  int num_row_pixels = 3;
  int num_rows = 2;
  size_t row_stride_bytes;
  unsigned char *d_samples;
  int *d_histogram[3];
  int num_levels[3] = {257, 257, 257};
  unsigned int lower_level[3] = {0, 0, 0};
  unsigned int upper_level[3] = {256, 256, 256};

  cudaStream_t S;
  cudaStreamCreate(&S);
  
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::MultiHistogramEven<4, 3>(d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels, lower_level, upper_level, num_row_pixels, num_rows, row_stride_bytes, S);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // CHECK: DPCT1026:{{.*}}: The call to cub::DeviceHistogram::MultiHistogramEven was removed because this call is redundant in SYCL.
  // CHECK: dpct::multi_histogram_even_roi<4, 3>(oneapi::dpl::execution::device_policy(*S), d_samples, d_histogram, num_levels, lower_level, upper_level, num_row_pixels, num_rows, row_stride_bytes);
  cub::DeviceHistogram::MultiHistogramEven<4, 3>(d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels, lower_level, upper_level, num_row_pixels, num_rows, row_stride_bytes, S);

  return true;
}

bool histgram_range_stream() {
  int num_samples = 10;
  float *d_samples;
  int *d_histogram;
  int num_levels = 7; // (seven level boundaries for six bins)
  float *d_levels;
  cudaStream_t S;
  cudaStreamCreate(&S);

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::HistogramRange(d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels, d_levels, num_samples, S);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // CHECK: DPCT1026:{{.*}}: The call to cub::DeviceHistogram::HistogramRange was removed because this call is redundant in SYCL.
  // CHECK: dpct::histogram_range(oneapi::dpl::execution::device_policy(*S), d_samples, d_histogram, num_levels, d_levels, num_samples);
  cub::DeviceHistogram::HistogramRange(d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels, d_levels, num_samples, S);
  return true;
}

bool histgram_range_roi_stream() {
  int num_row_samples = 5;
  int num_rows = 2;
  int row_stride_bytes = 7 * sizeof(float);
  float *d_samples;
  int *d_histogram;
  int num_levels = 7;
  float *d_levels;
  cudaStream_t S;
  cudaStreamCreate(&S);
  
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::HistogramRange(d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels, d_levels, num_row_samples, num_rows, row_stride_bytes, S);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // CHECK: DPCT1026:{{.*}}: The call to cub::DeviceHistogram::HistogramRange was removed because this call is redundant in SYCL.
  // CHECK: dpct::histogram_range_roi(oneapi::dpl::execution::device_policy(*S), d_samples, d_histogram, num_levels, d_levels, num_row_samples, num_rows, row_stride_bytes);
  cub::DeviceHistogram::HistogramRange(d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels, d_levels, num_row_samples, num_rows, row_stride_bytes, S);

  return true;
}

bool multi_histgram_range_stream() {
  int num_pixels = 5;
  unsigned char *d_samples;
  unsigned int *d_histogram[3];
  int num_levels[3] = {5, 5, 5};
  unsigned int *d_levels[3];

  cudaStream_t S;
  cudaStreamCreate(&S);
  
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::MultiHistogramRange<4, 3>(d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels, d_levels, num_pixels, S);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // CHECK: DPCT1026:{{.*}}: The call to cub::DeviceHistogram::MultiHistogramRange was removed because this call is redundant in SYCL.
  // CHECK: dpct::multi_histogram_range<4, 3>(oneapi::dpl::execution::device_policy(*S), d_samples, d_histogram, num_levels, d_levels, num_pixels);
  cub::DeviceHistogram::MultiHistogramRange<4, 3>(d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels, d_levels, num_pixels, S);

  return true;
}

bool multi_histgram_range_roi_stream() {
  int num_row_pixels = 3;
  int num_rows = 2;
  size_t row_stride_bytes;
  // clang-format off
  unsigned char *d_samples;
  // clang-format on
  int *d_histogram[3];
  int num_levels[3] = {5, 5, 5};
  unsigned int *d_levels[3];

  cudaStream_t S;
  cudaStreamCreate(&S);
  
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::MultiHistogramRange<4, 3>(d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels, d_levels, num_row_pixels, num_rows, row_stride_bytes, S);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // CHECK: DPCT1026:{{.*}}: The call to cub::DeviceHistogram::MultiHistogramRange was removed because this call is redundant in SYCL.
  // CHECK: dpct::multi_histogram_range_roi<4, 3>(oneapi::dpl::execution::device_policy(*S), d_samples, d_histogram, num_levels, d_levels, num_row_pixels, num_rows, row_stride_bytes);
  cub::DeviceHistogram::MultiHistogramRange<4, 3>(d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels, d_levels, num_row_pixels, num_rows, row_stride_bytes, S);

  return true;
}
// clang-format on
