// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/devicelevel/device_spmv %S/device_spmv.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/devicelevel/device_spmv/device_spmv.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/devicelevel/device_spmv/device_spmv.dp.cpp -o %T/devicelevel/device_spmv/device_spmv.dp.o %}

// CHECK: #include <oneapi/dpl/execution>
// CHECK: #include <oneapi/dpl/algorithm>
// CHECK: #include <dpct/sparse_utils.hpp>

#include <cub/cub.cuh>
#include <initializer_list>

template <class T> T *init(std::initializer_list<T> L) {
  T *Ptr = nullptr;
  cudaMallocManaged(&Ptr, sizeof(T) * L.size());
  std::copy(L.begin(), L.end(), Ptr);
  return Ptr;
}

int main() {
  int num_rows = 9;
  int num_cols = 9;
  int num_nonzeros = 24;
  float *d_values = init<float>(
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});

  int *d_column_indices = init(
      {1, 3, 0, 2, 4, 1, 5, 0, 4, 6, 1, 3, 5, 7, 2, 4, 8, 3, 7, 4, 6, 8, 5, 7});

  int *d_row_offsets = init({0, 2, 5, 7, 10, 14, 17, 19, 22, 24});

  float *d_vector_x = init<float>({1, 1, 1, 1, 1, 1, 1, 1, 1});
  float *d_vector_y = init<float>({0, 1, 0, 0, 0, 0, 0, 0, 0});

  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  // CHECK: DPCT1026:{{.*}} The call to cub::DeviceSpmv::CsrMV was removed because this functionality is redundant in SYCL.
  cub::DeviceSpmv::CsrMV(d_temp_storage, temp_storage_bytes, d_values,
                         d_row_offsets, d_column_indices, d_vector_x,
                         d_vector_y, num_rows, num_cols, num_nonzeros);

  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // CHECK: dpct::sparse::csrmv(q_ct1, d_values, d_row_offsets, d_column_indices, d_vector_x, d_vector_y, num_rows, num_cols);
  cub::DeviceSpmv::CsrMV(d_temp_storage, temp_storage_bytes, d_values,
                         d_row_offsets, d_column_indices, d_vector_x,
                         d_vector_y, num_rows, num_cols, num_nonzeros);
  cudaDeviceSynchronize();
  for (int i = 0; i < 9; ++i) {
    printf("%.2f%c", d_vector_y[i], (i == 8 ? '\n' : ' '));
  }

  cudaStream_t S;
  cudaStreamCreate(&S);
  d_temp_storage = NULL;
  temp_storage_bytes = 0;
  // CHECK: DPCT1026:{{.*}} The call to cub::DeviceSpmv::CsrMV was removed because this functionality is redundant in SYCL.
  cub::DeviceSpmv::CsrMV(d_temp_storage, temp_storage_bytes, d_values,
                         d_row_offsets, d_column_indices, d_vector_x,
                         d_vector_y, num_rows, num_cols, num_nonzeros, S);

  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // CHECK: dpct::sparse::csrmv(*S, d_values, d_row_offsets, d_column_indices, d_vector_x, d_vector_y, num_rows, num_cols);
  cub::DeviceSpmv::CsrMV(d_temp_storage, temp_storage_bytes, d_values,
                         d_row_offsets, d_column_indices, d_vector_x,
                         d_vector_y, num_rows, num_cols, num_nonzeros, S);
  cudaDeviceSynchronize();
  return 0;
}
