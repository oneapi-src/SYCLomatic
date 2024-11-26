// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/blocklevel/blockreduce_p4 %S/blockreduce_p4.cu --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file %T/blocklevel/blockreduce_p4/blockreduce_p4.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/blocklevel/blockreduce_p4/blockreduce_p4.dp.cpp -o %T/blocklevel/blockreduce_p4/blockreduce_p4.dp.o %}

#include <iostream>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

__global__ void foo() {
  float v = 0.0f;
  typedef cub::BlockReduce<float, 1024> BlockReduce;
  __shared__ typename BlockReduce::TempStorage m;
  // CHECK: v = sycl::reduce_over_group(item_ct1.get_group(), (item_ct1.get_group().get_local_linear_id() < item_ct1.get_local_range(2)) ? v : sycl::known_identity_v<sycl::plus<>, float>, sycl::plus<>());
  v = BlockReduce(m).Reduce(v, cub::Sum{}, blockDim.x);
}
