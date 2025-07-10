// Option: --use-experimental-features=bindless_images
#include <cuda.h>
void test() {
  CUfilter_mode fm = CU_TR_FILTER_MODE_POINT;
  CUtexref texRef;
  // Start
  cuTexRefGetMipmapFilterMode(&fm /*CUfilter_mode **/, texRef /*CUtexref*/);
  // End
}
