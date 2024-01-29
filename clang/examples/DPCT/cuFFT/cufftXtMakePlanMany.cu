#include "cufft.h"
#include "cufftXt.h"

void test(cufftHandle plan, int dim, long long int *n, long long int *inembed,
          long long int istride, long long int idist, cudaDataType itype,
          long long int *onembed, long long int ostride, long long int odist,
          cudaDataType otype, long long int num_of_trans, size_t *worksize,
          cudaDataType exectype) {
  // Start
  cufftXtMakePlanMany(plan /*cufftHandle*/, dim /*int*/, n /*long long int **/,
                      inembed /*long long int **/, istride /*long long int*/,
                      idist /*long long int*/, itype /*cudaDataType*/,
                      onembed /*long long int **/, ostride /*long long int*/,
                      odist /*long long int*/, otype /*cudaDataType*/,
                      num_of_trans /*long long int*/, worksize /*size_t **/,
                      exectype /*cudaDataType*/);
  // End
}
