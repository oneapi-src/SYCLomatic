// RUN: dpct --format-range=none --use-experimental-features=bindless_images -out-root %T/texture/surface_object_bindless_image %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck --input-file %T/texture/surface_object_bindless_image/surface_object_bindless_image.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/texture/surface_object_bindless_image/surface_object_bindless_image.dp.cpp -o %T/texture/surface_object_bindless_image/surface_object_bindless_image.dp.o %}

int main() {
  cudaSurfaceObject_t surf;
  cudaResourceDesc resDesc;
  // CHECK: surf = dpct::experimental::create_bindless_image(resDesc);
  cudaCreateSurfaceObject(&surf, &resDesc);
  // CHECK: dpct::experimental::destroy_bindless_image(surf, q_ct1);
  cudaDestroySurfaceObject(surf);
  // CHECK: resDesc = dpct::experimental::get_data(surf);
  cudaGetSurfaceObjectResourceDesc(&resDesc, surf);
}
