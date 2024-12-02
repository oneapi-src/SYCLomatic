// RUN: dpct --format-range=none -out-root %T/surface %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck --input-file %T/surface/surface.dp.cpp --match-full-lines %s

// CHECK: template<typename T> void kernel(dpct::image_accessor_ext<dpct_placeholder/*Fix the type manually*/, 1> surf) {
template<typename T> __global__ void kernel(cudaSurfaceObject_t surf) {
  int i;
  float j, k, l, m;
  // CHECK: surf.read_byte(i);
  surf1Dread<T>(surf, i);
  // CHECK: i = surf.read_byte(i);
  surf1Dread<T>(&i, surf, i);
  // CHECK: surf.read_byte(j, i);
  surf2Dread<T>(surf, j, i);
  // CHECK: i = surf.read_byte(j, i);
  surf2Dread<T>(&i, surf, j, i);
  // CHECK: surf.read_byte(k, j, i);
  surf3Dread<T>(surf, k, j, i);
  // CHECK: i = surf.read_byte(k, j, i);
  surf3Dread<T>(&i, surf, k, j, i);
}

static texture<uint2, 1> tex21;

__device__ void device01() {
  tex1D(tex21, 1.0f);
}
int main() {
  // CHECK: dpct::image_wrapper_base_p surf;
  cudaSurfaceObject_t surf;
  // CHECK: dpct::image_data resDesc;
  cudaResourceDesc resDesc;
  // CHECK: surf = dpct::create_image_wrapper(resDesc);
  cudaCreateSurfaceObject(&surf, &resDesc);

  kernel<int><<<1,1>>>(surf);
  cudaDestroySurfaceObject(surf);
  cudaGetSurfaceObjectResourceDesc(&resDesc, surf);
}
