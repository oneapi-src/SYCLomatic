#include "cusparse.h"

void test(cusparseDnMatDescr_t desc) {
  // Start
  void *value;
  cusparseDnMatGetValues(desc /*cusparseDnMatDescr_t*/, &value /*void ***/);
  // End
}
