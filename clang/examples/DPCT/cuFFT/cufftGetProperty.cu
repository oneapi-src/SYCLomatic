#include "cufft.h"

void test(libraryPropertyType type, int *value) {
  // Start
  cufftGetProperty(type /*libraryPropertyType*/, value /*int **/);
  // End
}
