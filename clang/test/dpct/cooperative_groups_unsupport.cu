// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-12.0, cuda-12.1
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v12.0, v12.1
// RUN: dpct --format-range=none -out-root %T/cooperative_groups_unsupport %s --cuda-include-path="%cuda-path/include" --use-experimental-features=logical-group --extra-arg="-std=c++14"
// RUN: FileCheck %s --match-full-lines --input-file %T/cooperative_groups_unsupport/cooperative_groups_unsupport.dp.cpp

#define _CG_ABI_EXPERIMENTAL
#include "cooperative_groups.h"
namespace cg = cooperative_groups;

__global__ void foo1() {
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1007:{{[0-9]+}}: Migration of this_thread_block is not supported.
//CHECK-NEXT:*/
//CHECK-NEXT:cg::thread_block tb = cg::experimental::this_thread_block(mem);
  cg::experimental::block_tile_memory<1, 1> mem;
  cg::thread_block tb = cg::experimental::this_thread_block(mem);
}

__global__ void foo2(cg::thread_block tb) {
//CHECK:/*
//CHECK-NEXT:DPCT1007:{{[0-9]+}}: Migration of tiled_partition is not supported.
//CHECK-NEXT:*/
//CHECK-NEXT:dpct::experimental::logical_group tbt64 = cg::experimental::tiled_partition<64>(tb);
  cg::thread_block_tile<64> tbt64 = cg::experimental::tiled_partition<64>(tb);
}
