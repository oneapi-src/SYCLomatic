#include "cusparse.h"

void test(cusparseDnMatDescr_t desc, void *value) {
  // Start
  cusparseDnMatSetValues(desc /*cusparseDnMatDescr_t*/, value /*void **/);
  // End
}
