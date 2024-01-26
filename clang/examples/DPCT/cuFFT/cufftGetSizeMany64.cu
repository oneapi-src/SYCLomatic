#include "cufft.h"

void test(cufftHandle plan, int dim, long long int *n, long long int *inembed,
          long long int istride, long long int idist, long long int *onembed,
          long long int ostride, long long int odist, cufftType type,
          long long int num_of_trans, size_t *worksize) {
  // Start
  cufftGetSizeMany64(plan /*cufftHandle*/, dim /*int*/, n /*long long int **/,
                     inembed /*long long int **/, istride /*long long int*/,
                     idist /*long long int*/, onembed /*long long int **/,
                     ostride /*long long int*/, odist /*long long int*/,
                     type /*cufftType*/, num_of_trans /*long long int*/,
                     worksize /*size_t **/);
  // End
}
