// UNSUPPORTED: cuda-8.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3
// UNSUPPORTED: v8.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3
// RUN: dpct --format-range=none -out-root %T/thrust-algo-supplement %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/thrust-algo-supplement/thrust-algo-supplement.dp.cpp --match-full-lines %s
#include <vector>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/extrema.h>
#include <thrust/gather.h>
#include <thrust/binary_search.h>

#include <thrust/find.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>



template<typename index_t>
void embedding_dense_backward_cuda_scan() {
  thrust::host_vector<int> h;
  thrust::device_vector<int> d;
  thrust::device_ptr<int> my_data;

  //CHECK: oneapi::dpl::inclusive_scan_by_segment(oneapi::dpl::execution::make_device_policy(q_ct1), oneapi::dpl::make_reverse_iterator(my_data), oneapi::dpl::make_reverse_iterator(my_data), oneapi::dpl::make_reverse_iterator(my_data), oneapi::dpl::make_reverse_iterator(my_data), oneapi::dpl::equal_tothrust::equal_to<index_t>(), oneapi::dpl::maximumthrust::maximum<index_t>());
  thrust::inclusive_scan_by_key(
    thrust::device,
    thrust::make_reverse_iterator(my_data),
    thrust::make_reverse_iterator(my_data),
    thrust::make_reverse_iterator(my_data),
    thrust::make_reverse_iterator(my_data),
    thrust::equal_to<index_t>(),
    thrust::maximum<index_t>()
  );

  //CHECK:oneapi::dpl::inclusive_scan_by_segment(oneapi::dpl::execution::make_device_policy(q_ct1), d.begin(), d.end(), d.begin(), d.end(), oneapi::dpl::equal_tothrust::equal_to<index_t>(), oneapi::dpl::maximumthrust::maximum<index_t>());
  thrust::inclusive_scan_by_key(
    thrust::device,
    d.begin(),
    d.end(),
    d.begin(),
    d.end(),
    thrust::equal_to<index_t>(),
    thrust::maximum<index_t>()
  );
}

template void embedding_dense_backward_cuda_scan<int>();
template void embedding_dense_backward_cuda_scan<int64_t>();
