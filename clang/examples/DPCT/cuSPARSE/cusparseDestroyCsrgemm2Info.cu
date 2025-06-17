#include "cusparse.h"

void test(csrgemm2Info_t info) {
  // Start
  cusparseDestroyCsrgemm2Info(info /*csrgemm2Info_t*/);
  // End
}
