#include "cusolverDn.h"

void test(cusolverDnParams_t params, cusolverDnFunction_t func,
          cusolverAlgMode_t algo) {
  // Start
  cusolverDnSetAdvOptions(params /*cusolverDnParams_t*/,
                          func /*cusolverDnFunction_t*/,
                          algo /*cusolverAlgMode_t*/);
  // End
}
