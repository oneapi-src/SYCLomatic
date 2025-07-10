// Option: --use-experimental-features=bindless_images

template <class T>
__global__ void test(T *ptr, int x, cudaSurfaceBoundaryMode boundary_mode,
                     cudaSurfaceObject_t obj) {
  // Start
  surf1Dread<T>(obj /*cudaSurfaceObject_t*/, x /*int*/,
                boundary_mode /*cudaSurfaceBoundaryMode*/);
  surf1Dread(ptr /*T **/, obj /*cudaSurfaceObject_t*/, x /*int*/,
             boundary_mode /*cudaSurfaceBoundaryMode*/);
  // End
}
