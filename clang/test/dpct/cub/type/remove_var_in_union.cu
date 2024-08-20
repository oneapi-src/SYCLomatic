// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct -in-root %S -out-root %T/type/remove_var_in_union %S/remove_var_in_union.cu --cuda-include-path="%cuda-path/include" -- -std=c++17 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/type/remove_var_in_union/remove_var_in_union.dp.cpp %s
// RUN: %if build_lit %{icpx -c -fsycl %T/type/remove_var_in_union/remove_var_in_union.dp.cpp -o %T/type/remove_var_in_union/remove_var_in_union.dp.o %}

#include <cub/cub.cuh>

// CHECK: template<int NUM_THREADS_PER_BLOCK>
// CHECK-NEXT: void kernel_dependent() {
// CHECK: }
template <int NUM_THREADS_PER_BLOCK>
__global__ void kernel_dependent() {
  typedef cub::BlockScan<int, NUM_THREADS_PER_BLOCK> BlockScan;
  typedef cub::BlockReduce<int, NUM_THREADS_PER_BLOCK> BlockReduce1;
  typedef cub::BlockReduce<double4, NUM_THREADS_PER_BLOCK> BlockReduce4;
  union TempStorage {
    typename BlockScan ::TempStorage for_scan;
    typename BlockReduce1::TempStorage for_reduce1;
    typename BlockReduce4::TempStorage for_reduce4;
  };
  __shared__ TempStorage smem_storage;
}

// CHECK: void kernel_no_dependent() {
// CHECK: }
__global__ void kernel_no_dependent() {
  typedef cub::BlockScan<int, 4> BlockScan;
  typedef cub::BlockReduce<int, 4> BlockReduce1;
  typedef cub::BlockReduce<double4, 4> BlockReduce4;

  union TempStorage {
    typename BlockScan ::TempStorage for_scan;
    typename BlockReduce1::TempStorage for_reduce1;
    typename BlockReduce4::TempStorage for_reduce4;
  };
  __shared__ TempStorage smem_storage;
}

// CHECK: void kernel_union_has_non_cubtype(uint8_t *smem_storage_ct1) {
// CHECK:  union TempStorage {
// CHECK:    int main;
// CHECK:  };
// CHECK:  TempStorage &smem_storage = *(TempStorage *)smem_storage_ct1;
// CHECK: }
template <int NUM_THREADS_PER_BLOCK>
__global__ void kernel_union_has_non_cubtype() {
  typedef cub::BlockScan<int, NUM_THREADS_PER_BLOCK> BlockScan;
  typedef cub::BlockReduce<int, NUM_THREADS_PER_BLOCK> BlockReduce1;
  typedef cub::BlockReduce<double4, NUM_THREADS_PER_BLOCK> BlockReduce4;
  union TempStorage {
    int main;
    typename BlockScan ::TempStorage for_scan;
    typename BlockReduce1::TempStorage for_reduce1;
    typename BlockReduce4::TempStorage for_reduce4;
  };
  __shared__ TempStorage smem_storage;
}

// CHECK: void kernel_union_name_not_tempstorage() {
// CHECK: }
template <int NUM_THREADS_PER_BLOCK>
__global__ void kernel_union_name_not_tempstorage() {
  typedef cub::BlockScan<int, NUM_THREADS_PER_BLOCK> BlockScan;
  typedef cub::BlockReduce<int, NUM_THREADS_PER_BLOCK> BlockReduce1;
  typedef cub::BlockReduce<double4, NUM_THREADS_PER_BLOCK> BlockReduce4;
  union StorageNameWithoutTemp {
    typename BlockScan ::TempStorage for_scan;
    typename BlockReduce1::TempStorage for_reduce1;
    typename BlockReduce4::TempStorage for_reduce4;
  };
  __shared__ StorageNameWithoutTemp smem_storage;
}

// CHECK: void other_kernel3(uint8_t *smem_storage_ct1) {
// CHECK-NEXT:   union TempStorage {
// CHECK-NEXT:     int a;
// CHECK-NEXT:     int b;
// CHECK-NEXT:     int d;
// CHECK-NEXT:   };
// CHECK-NEXT:   TempStorage &smem_storage = *(TempStorage *)smem_storage_ct1;
// CHECK-NEXT: }
template <int NUM_THREADS_PER_BLOCK>
__global__ void other_kernel3() {
  union TempStorage {
    int a;
    int b;
    int d;
  };
  __shared__ TempStorage smem_storage;
}
