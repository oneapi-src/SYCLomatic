// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0
// RUN: dpct --format-range=none -out-root %T/thrust-policy-2 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck %s --match-full-lines --input-file %T/thrust-policy-2/thrust-policy-2.dp.cpp

#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/functional.h>

class MyAlloctor {
public:
 typedef char value_type;
 char* allocate(std::ptrdiff_t size) { return nullptr; }
 void deallocate(char* p, size_t size) {}
};

template<typename T>
void foo() {
  cudaStream_t stream;
  MyAlloctor thrust_allocator;
  // CHECK: auto p = oneapi::dpl::execution::make_device_policy(*stream);
  auto p = thrust::cuda::par(thrust_allocator).on(stream);

  int num = 10;
  thrust::device_ptr<T> data_a;
  thrust::device_ptr<T> data_b;

  // CHECK: oneapi::dpl::inclusive_scan_by_segment(p, oneapi::dpl::make_reverse_iterator(data_a + num), oneapi::dpl::make_reverse_iterator(data_a), oneapi::dpl::make_reverse_iterator(data_b + num), oneapi::dpl::make_reverse_iterator(data_b + num), oneapi::dpl::equal_to<T>(), oneapi::dpl::maximum<T>());
  thrust::inclusive_scan_by_key(
    p,
    thrust::make_reverse_iterator(data_a + num),
    thrust::make_reverse_iterator(data_a),
    thrust::make_reverse_iterator(data_b + num),
    thrust::make_reverse_iterator(data_b + num),
    thrust::equal_to<T>(),
    thrust::maximum<T>()
  );
}

template void foo<int>();
