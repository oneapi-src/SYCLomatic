// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.2, cuda-11.4
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.2, v11.4
// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/DplExtrasAlgorithm/api_test20_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17
// RUN: grep "IsCalled" %T/DplExtrasAlgorithm/api_test20_out/MainSourceFiles.yaml | wc -l > %T/DplExtrasAlgorithm/api_test20_out/count.txt
// RUN: FileCheck --input-file %T/DplExtrasAlgorithm/api_test20_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DplExtrasAlgorithm/api_test20_out

// CHECK: 51
// TEST_FEATURE: DplExtrasAlgorithm_reduce_argmin

#include <cub/cub.cuh>
#include <initializer_list>

template <typename T> T *init(std::initializer_list<T> list) {
  T *p = nullptr;
  cudaMalloc<T>(&p, sizeof(T) * list.size());
  cudaMemcpy(p, list.begin(), sizeof(T) * list.size(), cudaMemcpyHostToDevice);
  return p;
}

int main() {
  // Declare, allocate, and initialize device-accessible pointers for input and
  // output
  int num_items = 7;
  int *d_in = init({8, 6, 7, 5, 3, 0, 9});
  cub::KeyValuePair<int, int> *d_out = init<cub::KeyValuePair<int, int>>({{-1, -1}});
  // ...
  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, d_out,
                         num_items);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run min-reduction
  cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, d_out,
                         num_items);
  
  cub::KeyValuePair<int, int> out;
  cudaMemcpy(&out, d_out, sizeof(out), cudaMemcpyDeviceToHost);
  printf("%d-%d\n", out.key, out.value);
  return 0;
}
