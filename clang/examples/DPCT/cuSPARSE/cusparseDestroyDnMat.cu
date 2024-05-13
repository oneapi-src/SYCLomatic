#include "cusparse.h"

void test(cusparseDnMatDescr_t desc) {
  // Start
  cusparseDestroyDnMat(desc /*cusparseDnMatDescr_t*/);
  // End
}
