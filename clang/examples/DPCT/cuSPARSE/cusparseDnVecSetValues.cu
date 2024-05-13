#include "cusparse.h"

void test(cusparseDnVecDescr_t desc, void *value) {
  // Start
  cusparseDnVecSetValues(desc /*cusparseDnVecDescr_t*/, value /*void **/);
  // End
}
