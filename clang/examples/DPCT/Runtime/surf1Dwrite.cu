// Option: --use-experimental-features=bindless_images

template <class T>
__global__ void test(T val, int x, cudaSurfaceBoundaryMode boundary_mode,
                     cudaSurfaceObject_t obj) {
  // Start
  surf1Dwrite(val /*T*/, obj /*cudaSurfaceObject_t*/, x /*int*/,
              boundary_mode /*cudaSurfaceBoundaryMode*/);
  // End
}
