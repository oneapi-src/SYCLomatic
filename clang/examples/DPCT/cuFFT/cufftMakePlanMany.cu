#include "cufft.h"

void test(cufftHandle plan, int dim, int *n, int *inembed, int istride,
          int idist, int *onembed, int ostride, int odist, cufftType type,
          int num_of_trans, size_t *worksize) {
  // Start
  cufftMakePlanMany(plan /*cufftHandle*/, dim /*int*/, n /*int **/,
                    inembed /*int **/, istride /*int*/, idist /*int*/,
                    onembed /*int **/, ostride /*int*/, odist /*int*/,
                    type /*cufftType*/, num_of_trans /*int*/,
                    worksize /*size_t **/);
  // End
}
