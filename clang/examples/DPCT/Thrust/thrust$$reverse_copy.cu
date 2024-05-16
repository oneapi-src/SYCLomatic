#include <vector>

#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
#include <thrust/partition.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>

void reverse_copy() {
  const int N = 6;
  int data[N] = {0, 1, 2, 3, 4, 5};
  thrust::device_vector<int> device_data(data, data + N);
  thrust::host_vector<int> host_data(data, data + N);
  thrust::device_vector<int> host_result(N);
  thrust::device_vector<int> device_result(N);
  int result[N];

  // Start
  /*1*/ thrust::reverse_copy(thrust::device, device_data.begin(),
                             device_data.end(), device_result.begin());
  /*2*/ thrust::reverse_copy(thrust::host, host_data.begin(), host_data.end(),
                             host_result.begin());
  /*3*/ thrust::reverse_copy(thrust::host, data, data + N, result);
  /*4*/ thrust::reverse_copy(device_data.begin(), device_data.end(),
                             device_result.begin());
  /*5*/ thrust::reverse_copy(host_data.begin(), host_data.end(),
                             host_result.begin());
  /*6*/ thrust::reverse_copy(data, data + N, result);
  // End
}
