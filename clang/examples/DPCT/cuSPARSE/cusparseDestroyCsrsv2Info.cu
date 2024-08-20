#include "cusparse.h"

void test(csrsv2Info_t info) {
  // Start
  cusparseDestroyCsrsv2Info(info /*csrsv2Info_t*/);
  // End
}
