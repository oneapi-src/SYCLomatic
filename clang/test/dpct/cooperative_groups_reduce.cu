// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0 
// RUN: dpct --format-range=none -out-root %T/cooperative_groups_reduce %s --cuda-include-path="%cuda-path/include" --extra-arg="-std=c++14"
// RUN: FileCheck %s --match-full-lines --input-file %T/cooperative_groups_reduce/cooperative_groups_reduce.dp.cpp


#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

__device__ void testReduce(double *sdata, const cg::thread_block &cta) {
  const unsigned int tid = cta.thread_rank();
  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
  // CHECK: DPCT1119:{{[0-9]+}}: cooperative_groups::__v1::thread_block_tile<16> has not supported yet, please try to migrate with the DPCT experimental option: --use-experimental-features=logical-group.
  cg::thread_block_tile<16> tile16 = cg::tiled_partition<16>(cta);
  int *idata;
  // CHECK: sycl::reduce_over_group(item_ct1.get_sub_group(), sdata[tid], sycl::plus<double>());
  cg::reduce(tile32, sdata[tid], cg::plus<double>());  
  // CHECK: sycl::reduce_over_group(item_ct1.get_sub_group(), sdata[tid], sycl::minimum<double>());
  cg::reduce(tile32, sdata[tid], cg::less<double>());  
  // CHECK: sycl::reduce_over_group(item_ct1.get_sub_group(), sdata[tid], sycl::maximum<double>());
  cg::reduce(tile32, sdata[tid], cg::greater<double>());  
  // CHECK: sycl::reduce_over_group(item_ct1.get_sub_group(), idata[tid], sycl::bit_and<int>());
  cg::reduce(tile32, idata[tid], cg::bit_and<int>());  
  // CHECK: sycl::reduce_over_group(item_ct1.get_sub_group(), idata[tid], sycl::bit_xor<int>());
  cg::reduce(tile32, idata[tid], cg::bit_xor<int>());
  // CHECK: sycl::reduce_over_group(item_ct1.get_sub_group(), idata[tid], sycl::bit_or<int>());
  cg::reduce(tile32, idata[tid], cg::bit_or<int>());
  cg::sync(cta);

}
