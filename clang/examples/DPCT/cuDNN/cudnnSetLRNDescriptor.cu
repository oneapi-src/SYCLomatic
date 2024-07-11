#include <cudnn.h>

void test(unsigned n, double alpha, double beta, double k) {
  // Start
  cudnnLRNDescriptor_t d;
  cudnnSetLRNDescriptor(d /*cudnnLRNDescriptor_t*/, n /*unsigned*/,
                        alpha /*double*/, beta /*double*/, k /*double*/);
  // End
}