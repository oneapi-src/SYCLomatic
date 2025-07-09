// Option: --use-experimental-features=bindless_images
#include <cuda.h>

void test() {
  CUtexref texRef;
  float min_clamp, max_clamp;
  // Start
  cuTexRefGetMipmapLevelClamp(&min_clamp/*float **/, &max_clamp/*float **/, texRef/*CUtexref*/);
  // End
}
