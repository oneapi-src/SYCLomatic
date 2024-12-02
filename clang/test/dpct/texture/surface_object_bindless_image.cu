// RUN: dpct --format-range=none --use-experimental-features=bindless_images -out-root %T/texture/surface_object_bindless_image %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck --input-file %T/texture/surface_object_bindless_image/surface_object_bindless_image.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/texture/surface_object_bindless_image/surface_object_bindless_image.dp.cpp -o %T/texture/surface_object_bindless_image/surface_object_bindless_image.dp.o %}

// CHECK: template<typename T> void kernel(sycl::ext::oneapi::experimental::sampled_image_handle surf) {
template<typename T> __global__ void kernel(cudaSurfaceObject_t surf) {
  int i;
  float j, k, l, m;
  // CHECK: dpct::experimental::sample_image_by_byte<T>(surf, float(i));
  surf1Dread<T>(surf, i);
  // CHECK: i = dpct::experimental::sample_image_by_byte<T>(surf, float(i));
  surf1Dread<T>(&i, surf, i);
  // CHECK: dpct::experimental::sample_image_by_byte<T>(surf, sycl::float2(j, i));
  surf2Dread<T>(surf, j, i);
  // CHECK: i = dpct::experimental::sample_image_by_byte<T>(surf, sycl::float2(j, i));
  surf2Dread<T>(&i, surf, j, i);
  // CHECK: dpct::experimental::sample_image_by_byte<T>(surf, sycl::float3(k, j, i));
  surf3Dread<T>(surf, k, j, i);
  // CHECK: i = dpct::experimental::sample_image_by_byte<T>(surf, sycl::float3(k, j, i));
  surf3Dread<T>(&i, surf, k, j, i);
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
