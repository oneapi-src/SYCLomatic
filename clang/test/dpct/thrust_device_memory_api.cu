// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct -out-root %T/thrust_device_memory_api %s --cuda-include-path="%cuda-path/include" --usm-level=none
// RUN: FileCheck --input-file %T/thrust_device_memory_api/thrust_device_memory_api.dp.cpp --match-full-lines %s

#include <thrust/device_delete.h>
#include <thrust/device_malloc.h>
#include <thrust/device_new.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

void foo() {

  const int N = 137;
  int val = 46;
  {
    // CHECK:    dpct::device_pointer<int> d_array = dpct::device_new<int>(N);
    // CHECK-NEXT:    dpct::device_delete(d_array, N);
    thrust::device_ptr<int> d_array = thrust::device_new<int>(N);
    thrust::device_delete(d_array, N);
  }

  {
    // CHECK:    dpct::device_pointer<int> d_mem = dpct::malloc_device<int>(N);
    // CHECK-NEXT:    dpct::device_pointer<int> d_array1 = dpct::device_new<int>(d_mem, N);
    // CHECK-NEXT:    dpct::device_pointer<int> d_array2 = dpct::device_new<int>(d_mem, val, N);
    // CHECK-NEXT:    dpct::device_delete(d_array1, N);
    // CHECK-NEXT:    dpct::device_delete(d_array2, N);
    thrust::device_ptr<int> d_mem = thrust::device_malloc<int>(N);
    thrust::device_ptr<int> d_array1 = thrust::device_new<int>(d_mem, N);
    thrust::device_ptr<int> d_array2 = thrust::device_new<int>(d_mem, val, N);
    thrust::device_delete(d_array1, N);
    thrust::device_delete(d_array2, N);
  }

  {
    const int N = 100;
    typedef thrust::pair<thrust::pointer<int, thrust::device_system_tag>, std::ptrdiff_t> ptr_and_size_t;
    thrust::device_system_tag device_sys;

    // CHECK:     ptr_and_size_t ptr_and_size = dpct::get_temporary_allocation<int>(device_sys, N);
    // CHECK-NEXT: dpct::release_temporary_allocation(device_sys, ptr_and_size.first, ptr_and_size.second);
    ptr_and_size_t ptr_and_size = thrust::get_temporary_buffer<int>(device_sys, N);
    thrust::return_temporary_buffer(device_sys, ptr_and_size.first, ptr_and_size.second);
  }

  {
    const int N = 100;
    thrust::device_system_tag device_sys;
    // CHECK: thrust::pointer<int, thrust::device_system_tag> ptr = dpct::malloc<int>(device_sys, N);
    // CHECK-NEXT: dpct::free(device_sys, ptr);
    thrust::pointer<int, thrust::device_system_tag> ptr = thrust::malloc<int>(device_sys, N);
    thrust::free(device_sys, ptr);
  }

  {
    const int N = 100;
    thrust::device_system_tag device_sys;
    // CHECK: thrust::pointer<void, thrust::device_system_tag> void_ptr = dpct::malloc(device_sys, N);
    // CHECK-NEXT:    dpct::free(device_sys, void_ptr);
    thrust::pointer<void, thrust::device_system_tag> void_ptr = thrust::malloc(device_sys, N);
    thrust::free(device_sys, void_ptr);
  }
}
