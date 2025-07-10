// Option: --use-experimental-features=bindless_images
#include <cuda.h>
void test() {
  CUtexref r;
  // Start
  cuTexRefDestroy(r/*CUtexref*/);
  // End
}