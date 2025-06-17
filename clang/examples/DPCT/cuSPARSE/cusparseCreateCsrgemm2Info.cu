#include "cusparse.h"

void test() {
  // Start
  csrgemm2Info_t info;
  cusparseCreateCsrgemm2Info(&info /*csrgemm2Info_t **/);
  // End
}
