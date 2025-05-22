
// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --use-experimental-features=local-memory-kernel-scope-allocation -format-range=none -in-root %S -out-root %T/cub_with_local_memory_kernel_allocation %S/cub_with_local_memory_kernel_allocation.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cub_with_local_memory_kernel_allocation/cub_with_local_memory_kernel_allocation.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/cub_with_local_memory_kernel_allocation/cub_with_local_memory_kernel_allocation.dp.cpp -o %T/cub_with_local_memory_kernel_allocation/cub_with_local_memory_kernel_allocation.dp.o %}

#include <cuda.h>
#include <cub/cub.cuh>

// CHECK:  template<typename T, int S>
// CHECK:  void kernel(T *A) {
// CHECK:      auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
// CHECK:      typedef dpct::group::group_load<T, 4, dpct::group::group_load_algorithm::sub_group_transpose, S> LoadFloat;
// CHECK:      union  type_ct1{
// CHECK:        typename LoadFloat::TempLocalMemory loadf;
// CHECK:        void *reducef;
// CHECK:      };
// CHECK:      auto &temp_storage = *sycl::ext::oneapi::group_local_memory_for_overwrite<type_ct1>(sycl::ext::oneapi::this_work_item::get_work_group<3>());
// CHECK:      T vals[4];
// CHECK:      LoadFloat(temp_storage.loadf).load(item_ct1, &(A[0]), vals, 10);
// CHECK:      auto &load = *sycl::ext::oneapi::group_local_memory_for_overwrite<typename LoadFloat::TempLocalMemory>(sycl::ext::oneapi::this_work_item::get_work_group<3>());
// CHECK:      LoadFloat(load).load(item_ct1, &(A[0]), vals, 10);
// CHECK:  }

template<typename T, int S>
__global__ void kernel(T *A) {
    typedef cub::BlockLoad<T, S, 4, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadFloat;
    typedef cub::BlockReduce<float, 32> BlockReduce;

    __shared__ union {
      typename LoadFloat::TempStorage loadf;
      typename BlockReduce::TempStorage reducef;
    } temp_storage;

    T vals[4];

    LoadFloat(temp_storage.loadf).Load(&(A[0]), vals, 10);

    __shared__ typename LoadFloat::TempStorage load;

    LoadFloat(load).Load(&(A[0]), vals, 10);
}

// CHECK:  void foo() {
// CHECK:        typedef dpct::group::group_load<int, 4, dpct::group::group_load_algorithm::sub_group_transpose, 32> LoadFloat;
// CHECK:        union  type_ct2{
// CHECK:            typename LoadFloat::TempLocalMemory loadf;
// CHECK:        };
// CHECK:        auto &temp_storage = *sycl::ext::oneapi::group_local_memory_for_overwrite<type_ct2>(sycl::ext::oneapi::this_work_item::get_work_group<3>());
// CHECK:        int vals[4];
// CHECK:        LoadFloat(temp_storage.loadf).load(sycl::ext::oneapi::this_work_item::get_nd_item<3>(), vals, vals, 10);
// CHECK:        auto &loadf2 = *sycl::ext::oneapi::group_local_memory_for_overwrite<typename LoadFloat::TempLocalMemory>(sycl::ext::oneapi::this_work_item::get_work_group<3>());
// CHECK:  }
__global__ void foo() {
  typedef cub::BlockLoad<int, 32, 4, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadFloat;
  __shared__ union {
      typename LoadFloat::TempStorage loadf;
  } temp_storage;
  int vals[4];
  LoadFloat(temp_storage.loadf).Load(vals, vals, 10);

  __shared__ typename LoadFloat::TempStorage loadf2;

}
// CHECK:  int main() {
// CHECK:        sycl::device dev_ct1;
// CHECK:        sycl::queue q_ct1(dev_ct1, sycl::property_list{sycl::property::queue::in_order()});
// CHECK:        q_ct1.parallel_for(
// CHECK:          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
// CHECK:          [=](sycl::nd_item<3> item_ct1) {
// CHECK:            foo();
// CHECK:          });
// CHECK:        q_ct1.parallel_for(
// CHECK:          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
// CHECK:          [=](sycl::nd_item<3> item_ct1) {
// CHECK:            kernel<int, 32>(0);
// CHECK:          });
// CHECK:        return 0;
// CHECK:      }
int main() {
  foo<<<1, 1>>>();
  kernel<int, 32><<<1,1>>>(0);
  return 0;
}
