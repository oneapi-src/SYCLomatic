#include "cusparse.h"

void test(libraryPropertyType type) {
  // Start
  int value;
  cusparseGetProperty(type /*libraryPropertyType*/, &value /*int **/);
  // End
}
