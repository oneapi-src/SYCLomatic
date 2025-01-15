// Option: --use-experimental-features=bindless_images

void test(CUgraphicsResource r) {
  // Start
  cuGraphicsUnregisterResource(r /*CUgraphicsResource*/);
  // End
}
