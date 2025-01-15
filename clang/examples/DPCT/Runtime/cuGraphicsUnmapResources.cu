// Option: --use-experimental-features=bindless_images

void test(int c, CUgraphicsResource *r, CUstream s) {
  // Start
  cuGraphicsUnmapResources(c /*int*/,
                           r /*CUgraphicsResource **/,
                           s /*CUstream*/);
  // End
}
