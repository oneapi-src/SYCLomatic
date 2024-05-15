#include "cusparse.h"

void test(cusparseSpMatDescr_t desc, cusparseSpMatAttribute_t attr, void *data,
          size_t data_size) {
  // Start
  cusparseSpMatGetAttribute(desc /*cusparseSpMatDescr_t*/,
                            attr /*cusparseSpMatAttribute_t*/, data /*void **/,
                            data_size /*size_t*/);
  // End
}
