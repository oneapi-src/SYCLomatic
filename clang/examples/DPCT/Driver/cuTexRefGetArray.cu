// Option: --use-experimental-features=bindless_images
#include <cuda.h>
void test() {
  CUtexref r;
  CUarray a;
  // Start
  cuTexRefGetArray(&a/*CUarray **/, r/*CUtexref*/);
  // End
}