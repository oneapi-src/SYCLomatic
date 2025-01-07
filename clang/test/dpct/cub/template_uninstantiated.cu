// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct -in-root %S -out-root %T/template_uninstantiated %S/template_uninstantiated.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/template_uninstantiated/template_uninstantiated.dp.cpp --match-full-lines %s

#include <cub/cub.cuh>


// CHECK: template <typename T>
// CHECK: void kernel() {
// CHECK:   typedef sycl::group<3> BS;
// CHECK-NOT: typename BS::TempStorage temp_storage;
// CHECK:   T thread_data, output;
// CHECK:   /*
// CHECK:   DPCT1028:{{[0-9]+}}: The ExclusiveSum member function call was not migrated because the caller function may not instantiated.
// CHECK:   */
// CHECK:   BS(temp_storage).ExclusiveSum(thread_data, output);
// CHECK: }
template <typename T>
__global__ void kernel() {

  typedef cub::BlockScan<T, 128> BS;

  __shared__ typename BS::TempStorage temp_storage;

  T thread_data, output;

  BS(temp_storage).ExclusiveSum(thread_data, output);
}
