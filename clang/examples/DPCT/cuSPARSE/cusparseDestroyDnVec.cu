#include "cusparse.h"

void test(cusparseDnVecDescr_t desc) {
  // Start
  cusparseDestroyDnVec(desc /*cusparseDnVecDescr_t*/);
  // End
}
