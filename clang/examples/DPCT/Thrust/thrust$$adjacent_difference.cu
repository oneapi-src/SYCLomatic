#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <thrust/adjacent_difference.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

void adjacent_difference_test() {
  // clang-format off
  // Start
  float *host_ptr_A;
  float *host_ptr_R;
  float *device_ptr_A;
  float *device_ptr_R;
  /*1*/ thrust::adjacent_difference(thrust::device, device_ptr_A, device_ptr_A+10, device_ptr_R); 
  /*2*/ thrust::adjacent_difference(host_ptr_A, host_ptr_A+10, host_ptr_R, thrust::minus<float>());
  /*3*/ thrust::adjacent_difference(thrust::device, device_ptr_A, device_ptr_A+10, device_ptr_R, thrust::minus<float>());
  /*4*/ thrust::adjacent_difference(host_ptr_A, host_ptr_A+10, host_ptr_R);
  // End
  // clang-format on
}
