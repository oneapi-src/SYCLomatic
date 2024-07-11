#include "cusparse.h"

void test(cusparseSpMatDescr_t desc, void *row_ptr, void *col_ind,
          void *value) {
  // Start
  cusparseCsrSetPointers(desc /*cusparseSpMatDescr_t*/, row_ptr /*void **/,
                         col_ind /*void **/, value /*void **/);
  // End
}
