#include "cusparse.h"

void test(cusparseDnVecDescr_t desc) {
  // Start
  void *value;
  cusparseDnVecGetValues(desc /*cusparseDnVecDescr_t*/, &value /*void ***/);
  // End
}
