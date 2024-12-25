// RUN: dpct --format-range=none --use-experimental-features=bindless_images -out-root %T/texture/surface_object_bindless_image %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck --input-file %T/texture/surface_object_bindless_image/surface_object_bindless_image.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/texture/surface_object_bindless_image/surface_object_bindless_image.dp.cpp -o %T/texture/surface_object_bindless_image/surface_object_bindless_image.dp.o %}
#include <cuda_runtime.h>
#include <cuda.h>

#include <cuda.h>
#include <cuda_runtime.h>
// CHECK: template<typename T> void kernel(sycl::ext::oneapi::experimental::unsampled_image_handle surf, T data) {
template<typename T> __global__ void kernel(cudaSurfaceObject_t surf, T data) {
  int i;
  float j, k, l, z, m;
  // CHECK: sycl::ext::oneapi::experimental::write_image(surf, int(i / sizeof(data)), data);
  surf1Dwrite(data, surf, i);
  // CHECK: sycl::ext::oneapi::experimental::write_image(surf, sycl::int2(i / sizeof(data), j), data);
  surf2Dwrite(data, surf, i, j);
  // CHECK:   sycl::ext::oneapi::experimental::write_image(surf, sycl::int3(i / sizeof(data), j, z), data);
  surf3Dwrite(data, surf, i, j, z);

}

__global__ void kernelWriteToLayeredSurface(cudaSurfaceObject_t surface, int width, int height, int layers) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int layer = blockIdx.z;

    if (x < width && y < height && layer < layers) {
        uchar4 value = make_uchar4(255, 0, 0, 255);
        // CHECK: sycl::ext::oneapi::experimental::write_image_array(surface, sycl::int2(x * sizeof(sycl::uchar4) / sizeof(value), y), layer, value);
        surf2DLayeredwrite(value, surface, x * sizeof(uchar4), y, layer);
    }
}

// CHECK: template<typename T> void kernel2(sycl::ext::oneapi::experimental::unsampled_image_handle surf) {
template<typename T> __global__ void kernel2(cudaSurfaceObject_t surf) {
  int i;
  float j, k, l, m;
  // CHECK: dpct::experimental::fetch_image_by_byte<T>(surf, int(i));
  surf1Dread<T>(surf, i);
  // CHECK: i = dpct::experimental::fetch_image_by_byte<T>(surf, int(i));
  surf1Dread<T>(&i, surf, i);
  // CHECK: dpct::experimental::fetch_image_by_byte<T>(surf, sycl::int2(j, i));
  surf2Dread<T>(surf, j, i);
  // CHECK: i = dpct::experimental::fetch_image_by_byte<T>(surf, sycl::int2(j, i));
  surf2Dread<T>(&i, surf, j, i);
  // CHECK: dpct::experimental::fetch_image_by_byte<T>(surf, sycl::int3(k, j, i));
  surf3Dread<T>(surf, k, j, i);
  // CHECK: i = dpct::experimental::fetch_image_by_byte<T>(surf, sycl::int3(k, j, i));
  surf3Dread<T>(&i, surf, k, j, i);
}
void surface_driver_function() {
  // CHECK: sycl::ext::oneapi::experimental::unsampled_image_handle surf;
  CUsurfObject surf;
  // CHECK: dpct::image_data pResDesc;
  CUDA_RESOURCE_DESC pResDesc;
  // CHECK: surf = dpct::experimental::create_bindless_image(pResDesc);
  cuSurfObjectCreate(&surf, &pResDesc);
  // CHECK: dpct::experimental::destroy_bindless_image(surf, dpct::get_in_order_queue());
  cuSurfObjectDestroy(surf);
  // CHECK: pResDesc = dpct::experimental::get_data(surf);
  cuSurfObjectGetResourceDesc(&pResDesc, surf);
}
int main() {
  // CHECK: sycl::ext::oneapi::experimental::unsampled_image_handle surf;
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
