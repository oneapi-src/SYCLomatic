#include "cufft.h"

void test(int dim, int *n, int *inembed, int istride, int idist, int *onembed,
          int ostride, int odist, cufftType type, int num_of_trans) {
  // Start
  cufftHandle plan;
  cufftPlanMany(&plan /*cufftHandle **/, dim /*int*/, n /*int **/,
                inembed /*int **/, istride /*int*/, idist /*int*/,
                onembed /*int **/, ostride /*int*/, odist /*int*/,
                type /*cufftType*/, num_of_trans /*int*/);
  // End
}
