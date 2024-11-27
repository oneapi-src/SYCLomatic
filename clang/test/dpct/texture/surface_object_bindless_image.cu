// RUN: dpct --format-range=none --use-experimental-features=bindless_images -out-root %T/texture/surface_object_bindless_image %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck --input-file %T/texture/surface_object_bindless_image/surface_object_bindless_image.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/texture/surface_object_bindless_image/surface_object_bindless_image.dp.cpp -o %T/texture/surface_object_bindless_image/surface_object_bindless_image.dp.o %}

#include <cuda.h>
#include <cuda_runtime.h>
template<typename T> __global__ void kernel(cudaSurfaceObject_t surf, T data) {
  int i;
  float j, k, l, z, m;
  // CHECK: dpct::experimental::write_image_by_byte(surf, float(i), data);
  surf1Dwrite(data, surf, i);
  // CHECK: dpct::experimental::write_image_by_byte(surf, sycl::float2(i, j), data);
  surf2Dwrite(data, surf, i, j);
  // CHECK: dpct::experimental::write_image_by_byte(surf, sycl::float3(i, j, z), data);
  surf3Dwrite(data, surf, i, j, z);

}

int main() {
  // CHECK: sycl::ext::oneapi::experimental::sampled_image_handle surf;
  cudaSurfaceObject_t surf;
  // CHECK: dpct::image_data resDesc;
  cudaResourceDesc resDesc;
  // CHECK: surf = dpct::experimental::create_bindless_image(resDesc);
  cudaCreateSurfaceObject(&surf, &resDesc);
  // CHECK: dpct::experimental::destroy_bindless_image(surf, dpct::get_in_order_queue());
  cudaDestroySurfaceObject(surf);
  // CHECK: resDesc = dpct::experimental::get_data(surf);
  cudaGetSurfaceObjectResourceDesc(&resDesc, surf);
}
